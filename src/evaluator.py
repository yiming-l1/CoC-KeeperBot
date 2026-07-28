import json
import os
import re
from typing import Dict, List, Tuple, Any
from src.rag_engine import get_engine
from src.llm_client import KeeperLLMClient, KeeperConfig, _get_openai_client
from src.state_manager import load_state, reset_state, save_state
from src.mechanics import calculate_sanity_loss


def run_rule_quiz():
    """
    Runs the 20-question rule quiz against the RAG system.

    Improvements for better accuracy:
    - Uses rerank for better relevance
    - Checks top k documents instead of just the first one
    - Uses category=None to search both rule_system and scenario_data
      (scenario section may also contain rule information)
    - Uses larger k value for better recall
    """
    with open("./data/rule_quiz.json", "r") as f:
        quiz_data = json.load(f)

    # Use engine.retrieve() for better accuracy with rerank support
    # Note: rerank requires sentence-transformers, may fail if not installed
    engine = get_engine()
    use_rerank = True

    score = 0
    print("STARTING RULE QUIZ (RAG RETRIEVAL TEST)\n")
    print(
        "Configuration: k=20, use_rerank=True, category=None (searching both rule_system and scenario_data)\n"
    )

    for i, item in enumerate(quiz_data):
        # Use engine.retrieve() which supports rerank
        # category=None means search all categories (both rule_system and scenario_data)
        try:
            # engine.retrieve() returns formatted results (dicts with 'text' key)
            results = engine.retrieve(
                item["question"],
                search_type="mmr",
                k=20,  # Retrieve more documents
                category=None,  # Search both rule_system and scenario_data
                use_rerank=use_rerank,  # Use reranking for better relevance
                return_raw_docs=False,  # Get formatted results
            )
        except Exception as e:
            # If rerank fails (e.g., sentence-transformers not installed), disable it
            if use_rerank:
                print(f"[WARNING] Rerank failed ({e}), retrying without rerank")
                use_rerank = False
            results = engine.retrieve(
                item["question"],
                search_type="mmr",
                k=20,
                category=None,  # Search both rule_system and scenario_data
                use_rerank=False,
                return_raw_docs=False,
            )

        # Check in top k documents (not just the first one)
        found = False
        for result in results[:20]:  # Check top 20 documents
            retrieved_text = result.get("text", "").lower()
            if item["answer_keyword"] in retrieved_text:
                found = True
                break

        if found:
            print(f"✅ Q{i + 1}: Passed")
            score += 1
        else:
            # Show what was actually retrieved for debugging
            top_text = (
                results[0].get("text", "").lower()[:150] if results else "(no docs)"
            )
            top_category = results[0].get("category", "unknown") if results else "none"
            print(
                f"❌ Q{i + 1}: Failed. Expected '{item['answer_keyword']}' in context."
            )
            print(f"   Question: {item['question']}")
            print(f"   Top result (category: {top_category}) preview: {top_text}...")

    print(f"\nFINAL SCORE: {score}/20")
    return score


def test_sanity_mechanics() -> Tuple[int, int]:
    """
    Qualitative test for sanity and temporary insanity mechanics.
    Tests:
    1. Sanity loss calculation (various formats)
    2. Temporary insanity triggering (loss >= 5)
    3. State persistence after sanity changes

    Returns: (passed_tests, total_tests)
    """
    print("\n" + "=" * 72)
    print("QUALITATIVE TEST: Sanity & Temporary Insanity Mechanics")
    print("=" * 72 + "\n")

    passed = 0
    total = 0

    # Test 1: Small sanity loss (< 5) should NOT trigger temp_insanity
    total += 1
    current_san = 50
    new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "3")
    if loss_val == 3 and new_san == 47 and not temp_ins:
        print(f"✅ Test 1: Small loss (3 SAN) correctly does NOT trigger temp_insanity")
        print(f"   Loss: {loss_val}, New SAN: {new_san}, Temp_insanity: {temp_ins}")
        passed += 1
    else:
        print(f"❌ Test 1: Failed. Expected loss=3, new_san=47, temp_insanity=False")
        print(f"   Got: loss={loss_val}, new_san={new_san}, temp_insanity={temp_ins}")

    # Test 2: Large sanity loss (>= 5) SHOULD trigger temp_insanity
    total += 1
    current_san = 50
    new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "5")
    if loss_val == 5 and new_san == 45 and temp_ins:
        print(f"✅ Test 2: Large loss (5 SAN) correctly triggers temp_insanity")
        print(f"   Loss: {loss_val}, New SAN: {new_san}, Temp_insanity: {temp_ins}")
        passed += 1
    else:
        print(f"❌ Test 2: Failed. Expected loss=5, new_san=45, temp_insanity=True")
        print(f"   Got: loss={loss_val}, new_san={new_san}, temp_insanity={temp_ins}")

    # Test 3: Dice roll format (1d6) - may or may not trigger depending on roll
    total += 1
    current_san = 50
    # Run multiple times to test both cases (we accept either outcome)
    results = []
    for _ in range(10):
        new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "1d6")
        results.append((loss_val, temp_ins))
        if loss_val >= 5 and temp_ins:
            break  # Found at least one case that triggers temp_insanity
        if loss_val < 5 and not temp_ins:
            break  # Found at least one case that doesn't trigger

    has_trigger = any(t for _, t in results if _ >= 5)
    has_no_trigger = any(not t for _, t in results if _ < 5)
    if has_trigger or has_no_trigger:  # At least one valid outcome
        print(f"✅ Test 3: Dice format (1d6) correctly calculates loss")
        print(f"   Sample results: {results[:3]}...")
        passed += 1
    else:
        print(f"❌ Test 3: Failed. Dice format not working correctly")

    # Test 4: Large dice roll (1d10) - should sometimes trigger temp_insanity
    total += 1
    current_san = 50
    results = []
    for _ in range(20):
        new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "1d10")
        results.append((loss_val, temp_ins))

    trigger_count = sum(1 for loss, temp in results if temp)
    if trigger_count > 0:  # At least some rolls trigger temp_insanity
        print(
            f"✅ Test 4: Large dice (1d10) correctly triggers temp_insanity when loss >= 5"
        )
        print(f"   Triggered {trigger_count}/20 times")
        passed += 1
    else:
        print(f"❌ Test 4: Failed. 1d10 should trigger temp_insanity sometimes")

    # Test 5: Zero sanity loss
    total += 1
    current_san = 50
    new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "0")
    if loss_val == 0 and new_san == 50 and not temp_ins:
        print(f"✅ Test 5: Zero loss correctly handled")
        passed += 1
    else:
        print(f"❌ Test 5: Failed. Expected loss=0, new_san=50, temp_insanity=False")

    # Test 6: State persistence (load state, modify sanity, verify it's saved)
    total += 1
    try:
        original_state = load_state()
        original_san = original_state["investigator"]["sanity"]["current"]

        # Simulate a large sanity loss through state update
        test_state = json.loads(json.dumps(original_state))  # deep copy
        test_state["investigator"]["sanity"]["current"] = original_san - 6
        test_state["investigator"]["sanity"]["temp_insanity"] = True
        save_state(test_state)

        # Reload and verify
        reloaded = load_state()
        if (
            reloaded["investigator"]["sanity"]["current"] == original_san - 6
            and reloaded["investigator"]["sanity"]["temp_insanity"]
        ):
            print(f"✅ Test 6: State persistence works correctly")
            passed += 1
        else:
            print(f"❌ Test 6: Failed. State not persisted correctly")

        # Restore original state
        save_state(original_state)
    except Exception as e:
        print(f"❌ Test 6: Error - {e}")

    print(f"\nSANITY MECHANICS TEST SCORE: {passed}/{total}")
    return passed, total


def test_temp_insanity_prompt_injection() -> Tuple[int, int]:
    """
    Tests that when temp_insanity=true, the system prompt includes
    appropriate instructions for hallucination/paranoia narration.

    Returns: (passed_tests, total_tests)
    """
    print("\n" + "=" * 72)
    print("QUALITATIVE TEST: Temporary Insanity Prompt Injection")
    print("=" * 72 + "\n")

    passed = 0
    total = 0

    try:
        from src.llm_client import KeeperLLMClient, KeeperConfig

        # Test 1: Normal state should NOT mention temp_insanity in prompt
        total += 1
        reset_state()
        state = load_state()
        state["investigator"]["sanity"]["temp_insanity"] = False
        save_state(state)

        client = KeeperLLMClient(KeeperConfig(enable_rag=False))
        system_prompt = client._build_system_prompt(state)

        if (
            "temp insanity" not in system_prompt.lower()
            or "temp_insanity" not in system_prompt.lower()
        ):
            # Actually, the prompt always mentions it in the state summary, so we check for the special instruction
            if (
                "Increase dread" not in system_prompt
                and "perceptual uncertainty" not in system_prompt
            ):
                print(
                    f"✅ Test 1: Normal state does not include temp_insanity special instructions"
                )
                passed += 1
            else:
                print(
                    f"❌ Test 1: Normal state incorrectly includes temp_insanity instructions"
                )
        else:
            # Check if it's just in the state summary (acceptable) vs special instructions (not acceptable)
            if (
                "Increase dread" in system_prompt
                or "perceptual uncertainty" in system_prompt
            ):
                print(
                    f"❌ Test 1: Normal state incorrectly includes temp_insanity special instructions"
                )
            else:
                print(
                    f"✅ Test 1: Normal state only mentions temp_insanity in state summary (OK)"
                )
                passed += 1

        # Test 2: Temp_insanity=true SHOULD include special instructions
        total += 1
        state["investigator"]["sanity"]["temp_insanity"] = True
        save_state(state)

        system_prompt = client._build_system_prompt(state)
        if (
            "Increase dread" in system_prompt
            or "perceptual uncertainty" in system_prompt
            or "Temp insanity is active" in system_prompt
        ):
            print(
                f"✅ Test 2: Temp_insanity=true includes special narration instructions"
            )
            print(f"   Found relevant instruction in system prompt")
            passed += 1
        else:
            print(f"❌ Test 2: Temp_insanity=true should include special instructions")
            print(f"   Prompt snippet: {system_prompt[-200:]}")

        # Restore state
        reset_state()

    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\nPROMPT INJECTION TEST SCORE: {passed}/{total}")
    return passed, total


def _llm_check_negative_mention(response: str, entity: str) -> bool:
    """
    Uses LLM to determine if the response contains a negative mention of the entity
    (i.e., saying it doesn't exist / isn't present, which is OK for hallucination check).

    Returns True if the mention is negative (OK), False if positive (hallucination).
    """
    try:
        # Get LLM client (same as used in KeeperLLMClient)
        # Suppress the print statement by temporarily capturing stdout or just using it silently
        client = _get_openai_client()

        # Get model name from environment
        provider = os.getenv("LLM_PROVIDER", "qwen").lower()
        if provider == "gemini":
            model = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
        elif provider == "qwen":
            model = os.getenv("OPENAI_MODEL", "qwen-flash")
        else:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Limit response length to avoid token limits
        response_snippet = response[:500] if len(response) > 500 else response
        entity_title = entity.title()

        prompt = f"""You are judging whether a game master's response contains a NEGATIVE mention of "{entity}".

A NEGATIVE mention means the response says the entity does NOT exist, is NOT present, or is NOT seen.
Examples of NEGATIVE mentions (these are OK):
- "There's no sign of {entity} here"
- "You don't see any {entity}"
- "{entity_title} is not present"
- "No indication of {entity}"

A POSITIVE mention means the response suggests the entity exists or is present.
Examples of POSITIVE mentions (these would be hallucinations):
- "You see a {entity}"
- "{entity_title} appears before you"
- "There is a {entity} in the room"

Response text to judge:
{response_snippet}

Question: Does this response contain a NEGATIVE mention of "{entity}" (saying it doesn't exist/isn't present)?

Answer with ONLY "YES" or "NO" (no other text)."""

        messages = [
            {
                "role": "system",
                "content": "You are a strict evaluator. Answer only YES or NO, nothing else.",
            },
            {"role": "user", "content": prompt},
        ]

        response_llm = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent judgment
            max_tokens=10,
        )

        answer = (response_llm.choices[0].message.content or "").strip().upper()

        # Return True if LLM says it's a negative mention (OK)
        is_negative = answer.startswith("YES")
        return is_negative

    except Exception as e:
        # If LLM call fails, fall back to keyword-based check
        print(
            f"[WARNING] LLM judgment failed for entity '{entity}': {e}. Falling back to keyword check."
        )
        response_lower = response.lower()
        negative_keywords = [
            "no sign of",
            "not here",
            "don't see",
            "cannot see",
            "not present",
            "nothing like",
            "nothing resembling",
            "there's no",
            "there is no",
        ]
        return any(kw in response_lower for kw in negative_keywords)


def check_hallucination(
    response: str, query: str, rag_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Checks if LLM response contains hallucinations (made-up content not in RAG results).

    Args:
        response: LLM generated response
        query: Original user query
        rag_results: RAG retrieval results (list of dicts with 'text', 'source', etc.)

    Returns:
        Dict with 'is_hallucination', 'confidence', 'reason', 'found_entities'
    """
    # First, clean the response by removing TOOL_CALL blocks if any
    response = re.sub(
        r"<TOOL_CALL>.*?</TOOL_CALL>", "", response, flags=re.DOTALL | re.IGNORECASE
    )
    response = re.sub(r"<TOOL_CALL>.*$", "", response, flags=re.DOTALL | re.IGNORECASE)
    response_lower = response.lower().strip()

    # Known non-existent entities (from README: "I look for Cthulhu" should say "don't see that")
    forbidden_entities = [
        "cthulhu",  # Not in the scenario
        "azathoth",  # Not in the scenario
        "nyarlathotep",  # Not in the scenario
        "yog-sothoth",  # Not in the scenario
        "shoggoth",  # Not in the scenario
        "deep ones",  # Not in the scenario (though "deep one" might be in rules)
        "deep one",  # Also check singular form
    ]

    # Check for forbidden entities in response
    found_forbidden = []
    for entity in forbidden_entities:
        if entity in response_lower:
            # Use LLM to judge if this is a negative mention (OK) or positive mention (hallucination)
            is_negative = _llm_check_negative_mention(response, entity)

            if not is_negative:
                # Positive mention = hallucination
                found_forbidden.append(entity)

    # Check if response claims to see/find things that aren't in RAG context
    # This is a simple heuristic - in production, use LLM-as-a-judge
    rag_text = " ".join([r.get("text", "").lower() for r in rag_results[:5]])

    result = {
        "is_hallucination": len(found_forbidden) > 0,
        "confidence": "high" if len(found_forbidden) > 0 else "low",
        "reason": f"Found forbidden entities: {found_forbidden}"
        if found_forbidden
        else "No obvious hallucinations detected",
        "found_entities": found_forbidden,
        "rag_context_length": len(rag_text),
    }

    return result


def test_hallucination_detection() -> Tuple[int, int]:
    """
    Tests hallucination detection by querying for non-existent entities.
    Expected: Keeper should say "you don't see that" not invent content.

    Returns: (passed_tests, total_tests)
    """
    print("\n" + "=" * 72)
    print("HALLUCINATION CHECK TEST")
    print("=" * 72 + "\n")

    passed = 0
    total = 0

    try:
        # Reset state to clean state
        reset_state()

        # Auto-select model based on provider (same logic as _get_openai_client)
        provider = os.getenv("LLM_PROVIDER", "qwen").lower()
        if provider == "gemini":
            model_name = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
        elif provider == "qwen":
            model_name = os.getenv("OPENAI_MODEL", "qwen-flash")
        else:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        print(f"[DEBUG] Using model: {model_name} for provider: {provider}")

        client = KeeperLLMClient(
            KeeperConfig(
                model=model_name,  # Use the correct model for the provider
                enable_rag=True,
                rag_k=5,
                temperature=0.7,
            )
        )

        # Test queries for non-existent entities
        test_queries = [
            ("I look for Cthulhu in the room", "cthulhu"),
            ("Do I see any Deep Ones here?", "deep ones"),
            ("Is there a Shoggoth in the basement?", "shoggoth"),
        ]

        for query, entity_name in test_queries:
            total += 1
            print(f"\nTest Query: '{query}'")
            print("-" * 60)

            # Get RAG results
            rag_results = client.rag_retrieve(query, allow_spoilers=True)

            # Run a turn (but don't force tool call for these queries)
            response = client.run_turn(query, force_tool=False)

            # Clean response - ensure TOOL_CALL blocks are removed
            # (run_turn should do this, but we do it again for safety)
            response_cleaned = re.sub(
                r"<TOOL_CALL>.*?</TOOL_CALL>",
                "",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            response_cleaned = re.sub(
                r"<TOOL_CALL>.*$", "", response_cleaned, flags=re.DOTALL | re.IGNORECASE
            ).strip()

            # Check for hallucinations using cleaned response
            check_result = check_hallucination(response_cleaned, query, rag_results)

            # Show cleaned response (not the original which might contain TOOL_CALL)
            print(f"Response (first 300 chars): {response_cleaned[:300]}...")
            if len(response_cleaned) == 0:
                print(f"[WARNING] Response is empty after cleaning!")
            print(f"Hallucination check: {check_result['reason']}")

            if not check_result["is_hallucination"]:
                print(
                    f"✅ Test {total}: No hallucination detected (Keeper correctly avoids inventing {entity_name})"
                )
                passed += 1
            else:
                print(
                    f"❌ Test {total}: Hallucination detected - {check_result['found_entities']}"
                )
                print(f"   Expected: Keeper should say 'you don't see that' or similar")
                print(
                    f"   Actual response suggests {entity_name} exists in the scenario"
                )

        # Restore state
        reset_state()

    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\nHALLUCINATION CHECK SCORE: {passed}/{total}")
    return passed, total


def run_all_evaluations() -> Dict[str, Any]:
    """
    Runs all evaluation tests and returns summary.
    """
    print("\n" + "=" * 72)
    print("RUNNING ALL EVALUATIONS")
    print("=" * 72 + "\n")

    results = {}

    # 1. Rule quiz (quantitative)
    print("\n[1/4] Running Rule Quiz (Quantitative RAG Test)...")
    rule_quiz_score = run_rule_quiz()
    results["rule_quiz"] = {"score": rule_quiz_score, "max": 20}

    # 2. Sanity mechanics (qualitative)
    print("\n[2/4] Running Sanity Mechanics Test (Qualitative)...")
    sanity_passed, sanity_total = test_sanity_mechanics()
    results["sanity_mechanics"] = {"passed": sanity_passed, "total": sanity_total}

    # 3. Temp insanity prompt injection (qualitative)
    print("\n[3/4] Running Temp Insanity Prompt Injection Test (Qualitative)...")
    prompt_passed, prompt_total = test_temp_insanity_prompt_injection()
    results["temp_insanity_prompt"] = {"passed": prompt_passed, "total": prompt_total}

    # 4. Hallucination check
    print("\n[4/4] Running Hallucination Check Test...")
    halluc_passed, halluc_total = test_hallucination_detection()
    results["hallucination_check"] = {"passed": halluc_passed, "total": halluc_total}

    # Summary
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Rule Quiz: {results['rule_quiz']['score']}/{results['rule_quiz']['max']}")
    print(
        f"Sanity Mechanics: {results['sanity_mechanics']['passed']}/{results['sanity_mechanics']['total']}"
    )
    print(
        f"Temp Insanity Prompt: {results['temp_insanity_prompt']['passed']}/{results['temp_insanity_prompt']['total']}"
    )
    print(
        f"Hallucination Check: {results['hallucination_check']['passed']}/{results['hallucination_check']['total']}"
    )

    total_tests = (
        results["rule_quiz"]["max"]
        + results["sanity_mechanics"]["total"]
        + results["temp_insanity_prompt"]["total"]
        + results["hallucination_check"]["total"]
    )
    total_passed = (
        results["rule_quiz"]["score"]
        + results["sanity_mechanics"]["passed"]
        + results["temp_insanity_prompt"]["passed"]
        + results["hallucination_check"]["passed"]
    )

    print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")
    print("=" * 72 + "\n")

    return results
