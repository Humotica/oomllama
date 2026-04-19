#!/usr/bin/env python3
"""
TIBET Full-Stack Pipeline Demo
==============================

cortex (JIS classificatie) -> bifurcation (AES-256-GCM encryptie) -> tbz (packaging)

Laat zien:
1. Data wordt geclassificeerd via JIS (clearance 0-255)
2. Data wordt encrypted met AES-256-GCM + X25519 ECDH
3. Encrypted data wordt verpakt in .tza-style blocks met Ed25519 signatures
4. Retrieval: alleen met geldige JIS claim kun je decrypten
5. Zonder claim: dood materiaal

Gebruikt:
- tibet-cortex: JIS access control + TIBET tokens
- cryptography: AES-256-GCM + X25519 + HKDF (Python equivalent van Rust bifurcation)
- Ed25519: block signing (tbz-style)

Usage:
  python3 tibet_pipeline_demo.py                    # Interactive mode
  python3 tibet_pipeline_demo.py "Geheim document"  # Direct input
  python3 tibet_pipeline_demo.py --file doc.txt      # File input
  python3 tibet_pipeline_demo.py --batch             # Batch demo met meerdere bestanden
"""

import sys
import os
import json
import time
import hashlib
import struct
import zlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

# Crypto
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as Ed25519Priv, Ed25519PublicKey as Ed25519Pub
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hashes import SHA256

# Tibet Cortex
from tibet_cortex import (
    JisClaim, JisPolicy, JisGate,
    TibetToken, Provenance, content_hash, generate_keypair,
    Envelope, EnvelopeBlock, BlockType,
)


# ═══════════════════════════════════════════════════════════════
# ANSI Colors
# ═══════════════════════════════════════════════════════════════
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_RED  = "\033[41m"
    BG_GREEN = "\033[42m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}  ████████╗██╗██████╗ ███████╗████████╗
  ╚══██╔══╝██║██╔══██╗██╔════╝╚══██╔══╝
     ██║   ██║██████╔╝█████╗     ██║
     ██║   ██║██╔══██╗██╔══╝     ██║
     ██║   ██║██████╔╝███████╗   ██║
     ╚═╝   ╚═╝╚═════╝ ╚══════╝   ╚═╝{C.RESET}

  {C.BOLD}Full-Stack Pipeline Demo{C.RESET}
  {C.DIM}cortex (JIS) -> bifurcation (AES-256-GCM) -> tbz (Ed25519 blocks){C.RESET}
  {C.DIM}"Geen JIS claim = dood materiaal."{C.RESET}
""")


# ═══════════════════════════════════════════════════════════════
# Bifurcation Engine (Python equivalent of Rust AirlockBifurcation)
# ═══════════════════════════════════════════════════════════════
@dataclass
class SealedBlock:
    """Encrypted block — equivalent to Rust BifurcationResult::Sealed"""
    block_index: int
    clearance: int
    ephemeral_pub: bytes       # X25519 public key (32 bytes)
    nonce: bytes               # AES-GCM nonce (12 bytes)
    ciphertext: bytes          # AES-256-GCM encrypted data
    tag: bytes                 # GCM authentication tag (included in ciphertext)
    content_hash: str          # SHA-256 of plaintext
    sealed_by: str
    sealed_at: str

    def hex_preview(self, n=32) -> str:
        return self.ciphertext[:n].hex()

    def total_size(self) -> int:
        return len(self.ciphertext) + len(self.nonce) + len(self.ephemeral_pub)


class Bifurcation:
    """Python equivalent of AirlockBifurcation — X25519 + HKDF + AES-256-GCM"""

    def __init__(self):
        self.static_key = X25519PrivateKey.generate()
        self.static_pub = self.static_key.public_key()
        self._block_counter = 0

    def seal(self, plaintext: bytes, clearance: int, origin: str) -> SealedBlock:
        """Encrypt data — equivalent to engine.seal() in Rust"""
        # 1. Ephemeral X25519 keypair (per block)
        eph_key = X25519PrivateKey.generate()
        eph_pub = eph_key.public_key()

        # 2. ECDH shared secret
        shared = eph_key.exchange(self.static_pub)

        # 3. HKDF-SHA256 derive AES key
        info = f"tibet-bifurcation-{self._block_counter}".encode()
        aes_key = HKDF(
            algorithm=SHA256(), length=32,
            salt=None, info=info,
        ).derive(shared)

        # 4. AES-256-GCM encrypt
        nonce = os.urandom(12)
        aesgcm = AESGCM(aes_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # 5. Content hash
        h = hashlib.sha256(plaintext).hexdigest()

        block = SealedBlock(
            block_index=self._block_counter,
            clearance=clearance,
            ephemeral_pub=eph_pub.public_bytes_raw(),
            nonce=nonce,
            ciphertext=ciphertext,
            tag=b"",  # Tag is embedded in ciphertext by AESGCM
            content_hash=f"sha256:{h}",
            sealed_by=origin,
            sealed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._block_counter += 1
        return block

    def open(self, block: SealedBlock, claim_clearance: int) -> Optional[bytes]:
        """Decrypt data — requires sufficient clearance"""
        # Access control check
        if claim_clearance < block.clearance:
            return None

        # 1. Reconstruct ephemeral public key
        eph_pub = X25519PublicKey.from_public_bytes(block.ephemeral_pub)

        # 2. ECDH shared secret (using our static private key)
        shared = self.static_key.exchange(eph_pub)

        # 3. HKDF derive same AES key
        info = f"tibet-bifurcation-{block.block_index}".encode()
        aes_key = HKDF(
            algorithm=SHA256(), length=32,
            salt=None, info=info,
        ).derive(shared)

        # 4. AES-256-GCM decrypt
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(block.nonce, block.ciphertext, None)
        return plaintext


# ═══════════════════════════════════════════════════════════════
# TZA Block Store (Python equivalent of TBZ bundle engine)
# ═══════════════════════════════════════════════════════════════
@dataclass
class TzaBlock:
    """A signed .tza block — equivalent to a TBZ data block"""
    index: int
    block_type: str              # "manifest" or "data"
    compressed: bytes            # zlib compressed
    uncompressed_size: int
    ed25519_signature: bytes
    content_hash: str
    jis_level: int
    description: str
    created_at: str


class TzaStore:
    """In-memory .tza store — equivalent to TBZ bundle engine"""

    MAGIC = b"TBZ"

    def __init__(self):
        self._signing_key = Ed25519Priv.generate()
        self._verify_key = self._signing_key.public_key()
        self.blocks: list[TzaBlock] = []
        self._manifest_entries = []

    def add_block(self, data: bytes, jis_level: int, description: str) -> TzaBlock:
        """Add data block — compress + sign + store"""
        compressed = zlib.compress(data, 3)  # zstd level 3 equivalent
        content_hash = f"sha256:{hashlib.sha256(data).hexdigest()}"

        # Ed25519 sign the compressed data
        signature = self._signing_key.sign(compressed)

        block = TzaBlock(
            index=len(self.blocks),
            block_type="data",
            compressed=compressed,
            uncompressed_size=len(data),
            ed25519_signature=signature,
            content_hash=content_hash,
            jis_level=jis_level,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.blocks.append(block)
        self._manifest_entries.append({
            "index": block.index,
            "description": description,
            "jis_level": jis_level,
            "uncompressed_size": block.uncompressed_size,
            "compressed_size": len(compressed),
            "content_hash": content_hash,
        })
        return block

    def verify(self, block: TzaBlock) -> bool:
        """Verify Ed25519 signature on block"""
        try:
            self._verify_key.verify(block.ed25519_signature, block.compressed)
            return True
        except Exception:
            return False

    def extract(self, block: TzaBlock) -> bytes:
        """Decompress block data"""
        return zlib.decompress(block.compressed)

    def manifest_json(self) -> str:
        return json.dumps({
            "protocol": "TBZ",
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "signing_key": self._verify_key.public_bytes_raw().hex(),
            "blocks": self._manifest_entries,
        }, indent=2)

    def save(self, path: str):
        """Write .tza archive to disk"""
        with open(path, "wb") as f:
            # Magic bytes
            f.write(self.MAGIC)
            # Version
            f.write(struct.pack("<H", 1))
            # Block count
            f.write(struct.pack("<I", len(self.blocks)))
            # Manifest as block 0
            manifest_data = self.manifest_json().encode()
            manifest_compressed = zlib.compress(manifest_data, 3)
            f.write(struct.pack("<I", len(manifest_compressed)))
            f.write(manifest_compressed)
            # Data blocks
            for block in self.blocks:
                # Block header: index, jis_level, compressed_size, signature
                f.write(struct.pack("<I", block.index))
                f.write(struct.pack("<B", block.jis_level))
                f.write(struct.pack("<I", len(block.compressed)))
                f.write(struct.pack("<I", len(block.ed25519_signature)))
                f.write(block.ed25519_signature)
                f.write(block.compressed)
        return path


# ═══════════════════════════════════════════════════════════════
# Pipeline Demo
# ═══════════════════════════════════════════════════════════════

# JIS clearance level names (matching Rust ClearanceLevel)
CLEARANCE_NAMES = {
    0: ("Unclassified", C.GREEN),
    50: ("Restricted", C.YELLOW),
    100: ("Confidential", C.YELLOW),
    150: ("Secret", C.RED),
    200: ("Top Secret", C.RED + C.BOLD),
}

def clearance_label(level: int) -> str:
    for threshold in sorted(CLEARANCE_NAMES.keys(), reverse=True):
        if level >= threshold:
            name, color = CLEARANCE_NAMES[threshold]
            return f"{color}{name} ({level}){C.RESET}"
    return f"{level}"


def classify_data(data: bytes, label: str) -> tuple[int, JisPolicy]:
    """Auto-classify data based on content patterns"""
    text = data.decode("utf-8", errors="ignore").lower()

    # Pattern-based classification
    if any(w in text for w in ["top secret", "nucleair", "wapen", "classified"]):
        return 200, JisPolicy(min_clearance=200, allowed_roles=["operator", "director"],
                              allowed_departments=["security", "defense"])
    elif any(w in text for w in ["secret", "geheim", "confidentieel", "intern"]):
        return 150, JisPolicy(min_clearance=150, allowed_roles=["operator", "analyst"],
                              allowed_departments=["security", "engineering"])
    elif any(w in text for w in ["restricted", "beperkt", "privat", "personal"]):
        return 100, JisPolicy(min_clearance=100)
    elif any(w in text for w in ["internal", "draft", "concept"]):
        return 50, JisPolicy(min_clearance=50)
    else:
        return 0, JisPolicy.public()


def step_header(num: int, title: str, icon: str = ""):
    print(f"\n{C.BOLD}{C.CYAN}  {'─' * 60}{C.RESET}")
    print(f"  {C.BOLD}{icon} Stap {num}: {title}{C.RESET}")
    print(f"{C.CYAN}  {'─' * 60}{C.RESET}")


def run_pipeline(inputs: list[tuple[str, bytes]]):
    """Run the full pipeline for one or more inputs"""

    engine = Bifurcation()
    store = TzaStore()
    provenance = Provenance()
    token_priv, token_pub = generate_keypair()

    # Collect results for retrieval phase
    sealed_blocks = []
    policies = []

    # ── STAP 1: JIS Classificatie ──
    step_header(1, "JIS Classificatie (tibet-cortex)", "\U0001f50d")
    print(f"  {C.DIM}Analyseert inhoud, bepaalt clearance level (0-255){C.RESET}\n")

    for label, data in inputs:
        clearance, policy = classify_data(data, label)
        policies.append((clearance, policy))

        preview = data[:80].decode("utf-8", errors="replace")
        if len(data) > 80:
            preview += "..."

        print(f"  {C.BOLD}{label}{C.RESET}")
        print(f"    Input:      {C.DIM}\"{preview}\"{C.RESET}")
        print(f"    Grootte:    {len(data)} bytes")
        print(f"    Classificatie: {clearance_label(clearance)}")
        if policy.allowed_roles:
            print(f"    Rollen:     {', '.join(policy.allowed_roles)}")
        if policy.allowed_departments:
            print(f"    Afdelingen: {', '.join(policy.allowed_departments)}")
        print()

    # ── STAP 2: Bifurcation Encryptie ──
    step_header(2, "Airlock Bifurcatie (AES-256-GCM + X25519)", "\U0001f510")
    print(f"  {C.DIM}Elke block: ephemeral X25519 DH -> HKDF-SHA256 -> AES-256-GCM{C.RESET}")
    print(f"  {C.DIM}Zonder JIS claim is dit dood materiaal.{C.RESET}\n")

    for i, ((label, data), (clearance, _)) in enumerate(zip(inputs, policies)):
        t0 = time.perf_counter()
        block = engine.seal(data, clearance, "tibet-pipeline-demo")
        dt_us = (time.perf_counter() - t0) * 1_000_000

        sealed_blocks.append(block)

        # TIBET token voor deze actie
        token = TibetToken.create(
            erin=block.content_hash,
            erachter=f"seal:{label}",
            actor="root_idd",
            jis_level=clearance,
        )
        # Link to parent token
        if provenance.latest():
            token = token.with_parent(provenance.latest().token_id)
        token = token.sign(token_priv)
        provenance.append(token)

        print(f"  {C.BOLD}{label}{C.RESET}  ->  Block #{block.block_index}")
        print(f"    Clearance:    {clearance_label(block.clearance)}")
        print(f"    Ephemeral:    {block.ephemeral_pub.hex()[:32]}...")
        print(f"    Nonce:        {block.nonce.hex()}")
        print(f"    Ciphertext:   {block.hex_preview(24)}... ({len(block.ciphertext)} bytes)")
        print(f"    Content hash: {block.content_hash}")
        print(f"    Seal tijd:    {dt_us:.0f} us")
        print(f"    TIBET token:  {token.token_id[:24]}...")
        print()

    # ── STAP 3: TZA Packaging ──
    step_header(3, "TBZ Packaging (zlib + Ed25519 signatures)", "\U0001f4e6")
    print(f"  {C.DIM}Elke encrypted block: compressie + Ed25519 handtekening{C.RESET}")
    print(f"  {C.DIM}Signing key: {store._verify_key.public_bytes_raw().hex()[:32]}...{C.RESET}\n")

    for i, (block, (label, _)) in enumerate(zip(sealed_blocks, inputs)):
        # Pack the sealed block as bytes (nonce + ephemeral_pub + ciphertext)
        block_bytes = block.nonce + block.ephemeral_pub + block.ciphertext

        tza_block = store.add_block(block_bytes, block.clearance, label)

        ratio = len(tza_block.compressed) / len(block_bytes) * 100
        verified = store.verify(tza_block)

        print(f"  {C.BOLD}Block #{tza_block.index}: {label}{C.RESET}")
        print(f"    Origineel:    {len(block_bytes)} bytes (encrypted)")
        print(f"    Gecomprimeerd: {len(tza_block.compressed)} bytes ({ratio:.1f}%)")
        print(f"    JIS level:    {clearance_label(tza_block.jis_level)}")
        print(f"    Ed25519 sig:  {tza_block.ed25519_signature.hex()[:32]}...")
        print(f"    Verificatie:  {C.GREEN}VALID{C.RESET}" if verified else f"    Verificatie:  {C.RED}FAIL{C.RESET}")
        print()

    # Save .tza file
    output_dir = "/tmp/tibet-demo"
    os.makedirs(output_dir, exist_ok=True)
    tza_path = os.path.join(output_dir, "demo.tza")
    store.save(tza_path)
    tza_size = os.path.getsize(tza_path)

    print(f"  {C.GREEN}{C.BOLD}Opgeslagen: {tza_path} ({tza_size} bytes){C.RESET}")
    print(f"  {C.DIM}Manifest:{C.RESET}")
    for entry in json.loads(store.manifest_json())["blocks"]:
        print(f"    Block {entry['index']}: {entry['description']} "
              f"(JIS {entry['jis_level']}, {entry['compressed_size']}B)")

    # ── STAP 4: Opslag Overzicht ──
    step_header(4, "Wat ligt er opgeslagen?", "\U0001f4be")
    print(f"  {C.DIM}De .tza bevat ALLEEN encrypted + compressed + signed blocks.{C.RESET}")
    print(f"  {C.DIM}Plaintext bestaat NERGENS meer op disk.{C.RESET}\n")

    total_plain = sum(len(d) for _, d in inputs)
    total_encrypted = sum(len(b.ciphertext) for b in sealed_blocks)
    total_stored = tza_size

    print(f"  Plaintext totaal:   {total_plain:>8} bytes  {C.RED}(bestaat niet meer){C.RESET}")
    print(f"  Encrypted totaal:   {total_encrypted:>8} bytes")
    print(f"  .tza op disk:       {total_stored:>8} bytes  {C.GREEN}(dit is alles){C.RESET}")
    print(f"  TIBET tokens:       {len(provenance.chain):>8} tokens in chain")
    print()
    print(f"  {C.YELLOW}{C.BOLD}Zonder JIS claim is dit wat je ziet:{C.RESET}")
    for block in sealed_blocks[:2]:
        garbage = block.ciphertext[:48].hex()
        print(f"    {C.DIM}{garbage}...{C.RESET}")
    print(f"    {C.DIM}(willekeurige bytes, onleesbaar, onherleidbaar){C.RESET}")

    # ── STAP 5: Retrieval Pogingen ──
    step_header(5, "Retrieval — Wie mag er bij?", "\U0001f511")

    # Define test claims
    test_claims = [
        ("Stagiair (clearance 25)", JisClaim(actor="stagiair.aint", clearance=25, role="intern", department="public")),
        ("Analist (clearance 100)", JisClaim(actor="analist.aint", clearance=100, role="analyst", department="engineering")),
        ("Operator (clearance 200)", JisClaim(actor="operator.aint", clearance=200, role="operator", department="security")),
    ]

    for claim_label, claim in test_claims:
        print(f"\n  {C.BOLD}{claim_label}{C.RESET}")
        print(f"  Actor: {claim.actor}, Role: {claim.role}, Dept: {claim.department}")
        print()

        for i, ((label, original_data), (clearance, policy), block) in enumerate(
            zip(inputs, policies, sealed_blocks)
        ):
            # JIS Gate check
            verdict = JisGate.evaluate(claim, policy)

            if verdict.allowed:
                # Try to decrypt
                t0 = time.perf_counter()
                plaintext = engine.open(block, claim.clearance)
                dt_us = (time.perf_counter() - t0) * 1_000_000

                if plaintext and plaintext == original_data:
                    preview = plaintext[:60].decode("utf-8", errors="replace")
                    print(f"    {C.GREEN}OPEN{C.RESET}  {label} ({clearance_label(clearance)})")
                    print(f"           -> \"{preview}{'...' if len(plaintext) > 60 else ''}\"")
                    print(f"           {C.DIM}{dt_us:.0f} us, {len(plaintext)} bytes, integrity OK{C.RESET}")

                    # TIBET token for open
                    open_token = TibetToken.create(
                        erin=block.content_hash,
                        erachter=f"open:{label}",
                        actor=claim.actor,
                        jis_level=clearance,
                    )
                    if provenance.latest():
                        open_token = open_token.with_parent(provenance.latest().token_id)
                    open_token = open_token.sign(token_priv)
                    provenance.append(open_token)
                elif plaintext:
                    print(f"    {C.RED}CORRUPT{C.RESET}  {label} — decryptie OK maar data mismatch!")
                else:
                    print(f"    {C.RED}DENIED{C.RESET}  {label} — clearance te laag voor decryptie")
            else:
                reasons = ", ".join(d.reason.value for d in verdict.denials)
                print(f"    {C.RED}DENIED{C.RESET}  {label} ({clearance_label(clearance)})")
                print(f"           {C.DIM}Reden: {reasons}{C.RESET}")

    # ── STAP 6: TIBET Provenance Chain ──
    step_header(6, "TIBET Provenance Chain (audit trail)", "\U0001f4dc")
    print(f"  {C.DIM}Elke actie (seal/open) genereert een signed TIBET token.{C.RESET}")
    print(f"  {C.DIM}Tokens zijn gelinkt in een append-only chain.{C.RESET}\n")

    chain_valid = provenance.verify_chain()
    sigs_valid = provenance.verify_signatures(token_pub)

    for i, token in enumerate(provenance.chain):
        actor = token.eromheen.actor if hasattr(token.eromheen, 'actor') else "unknown"
        signed = token.is_signed()
        print(f"  {C.DIM}[{i}]{C.RESET} {token.token_id[:20]}... "
              f"{C.BOLD}{token.erachter}{C.RESET}  "
              f"actor={actor}  "
              f"jis={token.eromheen.jis_level}  "
              f"{'signed' if signed else 'unsigned'}")

    print()
    print(f"  Chain integriteit:  {C.GREEN}VALID{C.RESET}" if chain_valid else f"  Chain integriteit:  {C.RED}BROKEN{C.RESET}")
    print(f"  Handtekeningen:    {C.GREEN}VALID{C.RESET}" if sigs_valid else f"  Handtekeningen:    {C.RED}BROKEN{C.RESET}")
    print(f"  Totaal tokens:     {len(provenance.chain)}")

    # ── Summary ──
    print(f"""
{C.CYAN}  {'=' * 60}{C.RESET}
  {C.BOLD}TIBET Full-Stack Pipeline — Samenvatting{C.RESET}

  {C.BOLD}Classificatie:{C.RESET}  tibet-cortex JIS (6-dimensionaal)
  {C.BOLD}Encryptie:{C.RESET}      AES-256-GCM + X25519 ECDH + HKDF-SHA256
  {C.BOLD}Packaging:{C.RESET}      TBZ blocks (zlib + Ed25519 signatures)
  {C.BOLD}Provenance:{C.RESET}     TIBET tokens (Ed25519 signed, append-only chain)

  {C.BOLD}Data op disk:{C.RESET}   {tza_path}
  {C.BOLD}Grootte:{C.RESET}        {tza_size} bytes (encrypted + compressed + signed)
  {C.BOLD}Blocks:{C.RESET}         {len(store.blocks)} data blocks
  {C.BOLD}TIBET tokens:{C.RESET}   {len(provenance.chain)} in chain

  {C.YELLOW}Zonder geldige JIS claim = dood materiaal.{C.RESET}
  {C.YELLOW}Identity IS the key.{C.RESET}

  {C.DIM}#HashesFromHolland{C.RESET}
{C.CYAN}  {'=' * 60}{C.RESET}
""")


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    banner()

    args = sys.argv[1:]

    if "--batch" in args:
        # Batch demo met meerdere classificatieniveaus
        inputs = [
            ("Publiek rapport", b"Dit is een publiek rapport over AI trends in 2026. Vrij beschikbaar."),
            ("Internal draft", "DRAFT: Internal roadmap Q3 2026 -- niet delen buiten team.".encode()),
            ("Restricted klantdata", "RESTRICTED: Klant #4821 - Privat financieel overzicht. Personeelsgegevens beperkt.".encode()),
            ("Geheim protocol", "GEHEIM: Protocol X-7 beveiligingsprocedure. Alleen security team.".encode()),
            ("Top Secret defensie", "TOP SECRET: Nucleair afschrikking scenario Delta-9. CLASSIFIED defense materiaal.".encode()),
        ]
    elif "--file" in args:
        idx = args.index("--file")
        if idx + 1 < len(args):
            filepath = args[idx + 1]
            with open(filepath, "rb") as f:
                data = f.read()
            inputs = [(os.path.basename(filepath), data)]
        else:
            print("  Gebruik: --file <pad>")
            sys.exit(1)
    elif args and not args[0].startswith("-"):
        # Direct text input
        text = " ".join(args)
        inputs = [("Direct input", text.encode("utf-8"))]
    else:
        # Interactive mode
        print(f"  {C.BOLD}Kies een modus:{C.RESET}")
        print(f"  [1] Batch demo (5 bestanden, alle clearance levels)")
        print(f"  [2] Typ je eigen tekst")
        print(f"  [3] Laad een bestand")
        print()

        choice = input(f"  Keuze (1/2/3): ").strip()

        if choice == "2":
            text = input(f"\n  Voer tekst in: ").strip()
            if not text:
                text = "Test document — TIBET pipeline demo"
            inputs = [("Gebruikersinput", text.encode("utf-8"))]
        elif choice == "3":
            path = input(f"\n  Bestandspad: ").strip()
            with open(path, "rb") as f:
                data = f.read()
            inputs = [(os.path.basename(path), data)]
        else:
            inputs = [
                ("Publiek rapport", b"Dit is een publiek rapport over AI trends in 2026. Vrij beschikbaar."),
                ("Internal draft", "DRAFT: Internal roadmap Q3 2026 -- niet delen buiten team.".encode()),
                ("Restricted klantdata", "RESTRICTED: Klant #4821 - Privat financieel overzicht. Personeelsgegevens beperkt.".encode()),
                ("Geheim protocol", "GEHEIM: Protocol X-7 beveiligingsprocedure. Alleen security team.".encode()),
                ("Top Secret defensie", "TOP SECRET: Nucleair afschrikking scenario Delta-9. CLASSIFIED defense materiaal.".encode()),
            ]

    run_pipeline(inputs)


if __name__ == "__main__":
    main()
