from src.guardrails.policy import PUBLIC_EMAIL, find_pii


def test_clean_text():
    assert find_pii("I built the perception stack at WATonomous using ROS2.") == []


def test_phone_number():
    assert "phone" in find_pii("Call me at 519-555-1234 anytime.")
    assert "phone" in find_pii("My number is (519) 555-1234.")


def test_sin():
    assert "sin" in find_pii("It's 123-456-789 if you must know.")


def test_street_address_and_postal_code():
    hits = find_pii("I live at 42 King Street, N2L 3G1.")
    assert "street_address" in hits
    assert "postal_code" in hits


def test_api_key():
    assert "api_key" in find_pii("here's the key sk-abc123def456ghi789")
    assert "api_key" in find_pii("token ghp_aBcDeFgH123456789")


def test_public_email_allowed_others_blocked():
    assert find_pii(f"Reach me at {PUBLIC_EMAIL}.") == []
    assert "non_public_email" in find_pii("Email my dad at dad@example.com.")
