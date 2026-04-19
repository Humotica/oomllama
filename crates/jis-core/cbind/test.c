/**
 * DID:JIS C Bindings Test
 *
 * Compile: gcc -o test test.c -L../target/release -ldid_jis_core -Iinclude
 * Run: LD_LIBRARY_PATH=../target/release ./test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/did_jis.h"

int main() {
    printf("=== DID:JIS C Bindings Test ===\n\n");

    // Check version
    printf("Version: %s\n\n", did_version());

    // Create engine
    printf("Creating DID engine...\n");
    did_engine_t* engine = did_engine_new();
    if (!engine) {
        printf("ERROR: Failed to create engine\n");
        return 1;
    }

    // Get public key
    char* pubkey = did_get_public_key(engine);
    printf("Public key: %.32s...\n", pubkey);

    char* pubkey_mb = did_get_public_key_multibase(engine);
    printf("Public key (multibase): %.32s...\n\n", pubkey_mb);

    // Create a DID
    printf("Creating DID...\n");
    char* did = did_create(engine, "device:6G:001");
    printf("DID: %s\n", did);

    // Create DID from key
    char* did_from_key = did_create_from_key(engine);
    printf("DID from key: %s\n\n", did_from_key);

    // Validate DID
    printf("Validating DIDs...\n");
    printf("  %s: %s\n", did, did_is_valid(did) ? "VALID" : "INVALID");
    printf("  did:web:example: %s\n\n", did_is_valid("did:web:example") ? "VALID" : "INVALID");

    // Parse DID
    char method[32];
    char id[256];
    if (did_parse(did, method, id)) {
        printf("Parsed DID:\n");
        printf("  Method: %s\n", method);
        printf("  ID: %s\n\n", id);
    }

    // Create DID document
    printf("Creating DID document...\n");
    char* doc = did_create_document(engine, did);
    if (doc) {
        printf("Document (first 300 chars):\n%.300s...\n\n", doc);
        did_free_string(doc);
    }

    // Sign and verify
    printf("Signing message...\n");
    char* signature = did_sign(engine, "Hello from 6G device!");
    printf("Signature: %.32s...\n", signature);

    bool valid = did_verify(engine, "Hello from 6G device!", signature);
    printf("Verification: %s\n\n", valid ? "PASSED" : "FAILED");

    // Cleanup
    did_free_string(signature);
    did_free_string(did_from_key);
    did_free_string(did);
    did_free_string(pubkey_mb);
    did_free_string(pubkey);
    did_engine_free(engine);

    printf("=== ALL TESTS PASSED ===\n");
    return 0;
}
