/**
 * DID:JIS - Decentralized Identifiers for JTel Identity Standard
 * C Bindings for embedded systems, 6G, and hardware integration
 *
 * Copyright (c) 2026 Humotica
 * License: MIT OR Apache-2.0
 *
 * Example:
 *   #include <did_jis.h>
 *
 *   did_engine_t* engine = did_engine_new();
 *   const char* did = did_create(engine, "device:001");
 *   const char* doc = did_create_document(engine, did);
 *   did_free_string(doc);
 *   did_free_string(did);
 *   did_engine_free(engine);
 */

#ifndef DID_JIS_H
#define DID_JIS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to DID engine */
typedef struct did_engine did_engine_t;

/* ============================================
 * Engine Lifecycle
 * ============================================ */

/**
 * Create a new DID engine with fresh Ed25519 keypair
 * Returns: New engine handle, or NULL on failure
 * Caller must free with did_engine_free()
 */
did_engine_t* did_engine_new(void);

/**
 * Create engine from existing secret key
 * @param secret_hex: 64-character hex string (32 bytes)
 * Returns: New engine handle, or NULL on invalid key
 */
did_engine_t* did_engine_from_secret(const char* secret_hex);

/**
 * Free engine and associated resources
 */
void did_engine_free(did_engine_t* engine);

/* ============================================
 * Key Management
 * ============================================ */

/**
 * Get public key as hex string
 * Returns: 64-character hex string
 * Caller must free with did_free_string()
 */
char* did_get_public_key(const did_engine_t* engine);

/**
 * Get public key in multibase format
 * Returns: Multibase-encoded string
 * Caller must free with did_free_string()
 */
char* did_get_public_key_multibase(const did_engine_t* engine);

/* ============================================
 * DID Operations
 * ============================================ */

/**
 * Create a did:jis identifier
 * @param engine: DID engine handle
 * @param id: Identifier part (e.g., "alice", "device:001")
 * Returns: Full DID string (e.g., "did:jis:alice")
 * Caller must free with did_free_string()
 */
char* did_create(const did_engine_t* engine, const char* id);

/**
 * Create a DID from public key hash
 * @param engine: DID engine handle
 * Returns: DID based on key hash (e.g., "did:jis:a1b2c3d4...")
 * Caller must free with did_free_string()
 */
char* did_create_from_key(const did_engine_t* engine);

/**
 * Parse a DID string
 * @param did: DID to parse (e.g., "did:jis:alice")
 * @param method: Output buffer for method (at least 32 bytes)
 * @param id: Output buffer for id (at least 256 bytes)
 * Returns: true if valid, false otherwise
 */
bool did_parse(const char* did, char* method, char* id);

/**
 * Validate a did:jis identifier
 * @param did: DID to validate
 * Returns: true if valid did:jis, false otherwise
 */
bool did_is_valid(const char* did);

/* ============================================
 * DID Document Operations
 * ============================================ */

/**
 * Create a signed DID document (JSON)
 * @param engine: DID engine handle
 * @param did: DID for the document
 * Returns: JSON string of signed DID document
 * Caller must free with did_free_string()
 */
char* did_create_document(const did_engine_t* engine, const char* did);

/* ============================================
 * Signing Operations
 * ============================================ */

/**
 * Sign a message
 * @param engine: DID engine handle
 * @param message: Null-terminated string to sign
 * Returns: Hex-encoded signature
 * Caller must free with did_free_string()
 */
char* did_sign(const did_engine_t* engine, const char* message);

/**
 * Verify a signature
 * @param engine: DID engine handle
 * @param message: Original message
 * @param signature: Hex-encoded signature
 * Returns: true if valid, false otherwise
 */
bool did_verify(const did_engine_t* engine, const char* message, const char* signature);

/**
 * Verify with external public key
 * @param message: Original message
 * @param signature: Hex-encoded signature
 * @param public_key_hex: 64-character hex public key
 * Returns: true if valid, false otherwise
 */
bool did_verify_with_key(const char* message, const char* signature, const char* public_key_hex);

/* ============================================
 * Memory Management
 * ============================================ */

/**
 * Free a string returned by DID functions
 */
void did_free_string(char* s);

/* ============================================
 * Version Info
 * ============================================ */

/**
 * Get DID:JIS library version
 * Returns: Version string (e.g., "0.1.0")
 * Do NOT free this string
 */
const char* did_version(void);

#ifdef __cplusplus
}
#endif

#endif /* DID_JIS_H */
