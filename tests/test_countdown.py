from rhjl.countdown import evaluate, extract_answer, score

N, T = [3, 5, 7], 22


def c(ans, think="<think>\nhmm\n</think>\n\n"):
    return f"{think}<answer>{ans}</answer>"


def test_correct():
    assert score(c("3 * 5 + 7"), N, T)["correct"]
    assert score(c("7 + 5 * 3"), N, T)["correct"]


def test_wrong_value():
    s = score(c("3 + 5 + 7"), N, T)
    assert s["uses_nums"] and not s["hits_target"] and not s["correct"]


def test_number_reuse_or_omission():
    assert not score(c("3 * 7 + 1"), N, T)["correct"]   # extra literal
    assert not score(c("15 + 7"), N, T)["correct"]      # hits target but wrong numbers
    assert score(c("15 + 7"), N, T)["hits_target"]


def test_exact_division():
    assert score(c("(7 - 5) / 3 * 33"), [7, 5, 3, 33], 22)["correct"]
    assert evaluate("1/3*3")[0] == 1


def test_bad_syntax_and_safety():
    for bad in ["__import__('os')", "2**10", "3 // 2", "", "5 / 0", "abc"]:
        assert evaluate(bad) is None


def test_answer_in_thinking_ignored():
    comp = "<think>\n<answer>3 * 5 + 7</answer>\n</think>\n\nno answer here"
    assert extract_answer(comp) is None


def test_unfinished_thinking():
    assert extract_answer("<think>\nstill going <answer>3*5+7</answer>") is None


def test_last_answer_and_equals():
    comp = c("1+1") + " actually <answer>3 * 5 + 7 = 22</answer>"
    assert score(comp, N, T)["correct"]


def test_no_think_block():
    assert score("<answer>3*5+7</answer>", N, T)["correct"]


def test_thinking_prompt_completion():
    # with enable_thinking the prompt already holds "<think>\n"; completion has only the close tag
    assert score("reasoning...\n</think>\n\n<answer>3*5+7</answer>", N, T, thinking=True)["correct"]
    assert not score("still reasoning <answer>3*5+7</answer>", N, T, thinking=True)["formatted"]
