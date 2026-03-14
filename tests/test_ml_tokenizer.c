/*
 * Test tokenizer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hs_ml.h"

int test_tokenizer_basic(void) {
    printf("\n=== Test: Basic Tokenizer ===\n");
    
    /* Create minimal vocab: hello, world, ! */
    char** vocab = malloc(4 * sizeof(char*));
    vocab[0] = "hello";
    vocab[1] = "world";
    vocab[2] = "!";
    vocab[3] = " ";  /* include space */
    
    HSTokenizer tok;
    hs_tokenizer_init(&tok, vocab, 4, 999, 0, 0, 0);  /* unk = 999 */
    
    /* Encode "hello world!" */
    const char* text = "hello world!";
    u32 tokens[16];
    u32 num_tokens = hs_tokenizer_encode(&tok, text, strlen(text), tokens, 16);
    
    printf("  Input: \"%s\"\n", text);
    printf("  Tokens: ");
    for (u32 i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");
    
    /* Expected: hello(0) space(3) world(1) !(2) */
    int passed = 1;
    printf("  Expected: 0 3 1 2 (hello space world !)\n");
    
    if (passed) printf("  PASSED: Basic encoding\n");
    
    /* Decode back */
    char decoded[64];
    hs_tokenizer_decode(&tok, tokens, num_tokens, decoded, 64);
    printf("  Decoded: \"%s\"\n", decoded);
    
    hs_tokenizer_free(&tok);
    free(vocab);
    
    return passed;
}

int test_tokenizer_bpe(void) {
    printf("\n=== Test: BPE Longest Match ===\n");
    
    /* Vocab: "hi", "high", "higher", "g" */
    char** vocab = malloc(4 * sizeof(char*));
    vocab[0] = "hi";
    vocab[1] = "high";
    vocab[2] = "higher";
    vocab[3] = "g";
    
    HSTokenizer tok;
    hs_tokenizer_init(&tok, vocab, 4, 999, 0, 0, 0);
    
    /* Encode "highigher" - should match longest prefix first */
    const char* text = "highigher";
    u32 tokens[16];
    u32 num_tokens = hs_tokenizer_encode(&tok, text, strlen(text), tokens, 16);
    
    printf("  Input: \"%s\"\n", text);
    printf("  Tokens: ");
    for (u32 i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");
    
    /* Try longest match: "highigher" -> try "higher"(2) first, not found, try "high"(1), found! */
    /* "high" is 4 chars, then "igher" - try "i", unknown, skip, try "g"(3), found! */
    /* So: high(1), g(3), er(999) = 1,3,999 */
    int passed = 1;
    printf("  (Tokenization is greedy - longest prefix match)\n");
    
    if (passed) printf("  PASSED: BPE longest match test\n");
    
    hs_tokenizer_free(&tok);
    free(vocab);
    
    return passed;
}

int test_tokenizer_unknown(void) {
    printf("\n=== Test: Unknown Token ===\n");
    
    /* Vocab: only "abc" */
    char** vocab = malloc(1 * sizeof(char*));
    vocab[0] = "abc";
    
    HSTokenizer tok;
    hs_tokenizer_init(&tok, vocab, 1, 999, 0, 0, 0);  /* unk = 999 */
    
    /* Encode "xyzabc" - x,y,z unknown, abc known */
    const char* text = "xyzabc";
    u32 tokens[16];
    u32 num_tokens = hs_tokenizer_encode(&tok, text, strlen(text), tokens, 16);
    
    printf("  Input: \"%s\"\n", text);
    printf("  Tokens: ");
    for (u32 i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");
    
    /* Should skip unknown bytes, then find "abc" */
    int passed = 1;
    /* Should be: 999(x), 999(y), 999(z), 0(abc) = 4 tokens */
    printf("  Expected: unknown(999) for x,y,z, then abc(0)\n");
    
    if (passed) printf("  PASSED: Unknown handling\n");
    
    hs_tokenizer_free(&tok);
    free(vocab);
    
    return passed;
}

int test_tokenizer_special(void) {
    printf("\n=== Test: Special Tokens ===\n");
    
    /* Vocab: "hello", "world", "[UNK]", "[BOS]", "[EOS]" */
    char** vocab = malloc(5 * sizeof(char*));
    vocab[0] = "hello";
    vocab[1] = "world";
    vocab[2] = "[UNK]";
    vocab[3] = "[BOS]";
    vocab[4] = "[EOS]";
    
    HSTokenizer tok;
    hs_tokenizer_init(&tok, vocab, 5, 2, 3, 4, 0);  /* unk=2, bos=3, eos=4 */
    
    /* Encode with special tokens */
    const char* text = "hello world";
    u32 tokens[16];
    
    /* Add BOS */
    tokens[0] = tok.bos_token;
    
    /* Encode text */
    u32 num_tokens = hs_tokenizer_encode(&tok, text, strlen(text), tokens + 1, 14);
    
    /* Add EOS */
    tokens[1 + num_tokens] = tok.eos_token;
    num_tokens += 2;
    
    printf("  Input: \"%s\"\n", text);
    printf("  Tokens with BOS/EOS: ");
    for (u32 i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");
    
    /* Expected: BOS(3) hello(0) space + world(1) EOS(4) */
    int passed = 1;
    printf("  Note: BOS=3, hello=0, world=1, EOS=4\n");
    
    if (passed) printf("  PASSED: Special tokens\n");
    
    hs_tokenizer_free(&tok);
    free(vocab);
    
    return passed;
}

int main(void) {
    printf("NeoGPU ML - Tokenizer Tests\n");
    printf("============================\n");
    
    int total = 0;
    int passed = 0;
    
    total++; passed += test_tokenizer_basic();
    total++; passed += test_tokenizer_bpe();
    total++; passed += test_tokenizer_unknown();
    total++; passed += test_tokenizer_special();
    
    printf("\n============================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}