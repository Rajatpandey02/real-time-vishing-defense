# scam_text.py
# Lightweight scam-language classifier for call transcripts (TF-IDF + logistic regression).
# The built-in examples are SELF-WRITTEN and only a starting point: add 100+ more lines of
# each class (scam-style and normal call-style) to improve it, and say in the report that
# the text data is synthetic.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SCAM = [
    "This is your bank. Your account will be blocked today unless you share the OTP now.",
    "Please transfer the amount immediately, it is an emergency and you must not tell anyone.",
    "Sir, your card is compromised. Read out the six digit code that was just sent to you.",
    "I am calling from the tax department, you must pay the penalty right now or face arrest.",
    "Beta, I am in trouble, send money to this number urgently, do not call anyone else.",
    "Your KYC is expired. Share your Aadhaar and card details to avoid account suspension.",
    "You have won a prize. Pay the processing fee today to receive your reward.",
    "This is customer support. Install this app so I can fix your account and see your screen.",
    "Your package is held by customs. Send a payment immediately or it will be destroyed.",
    "Do not hang up. Confirm your PIN so we can stop the fraudulent transaction.",
    "Your SIM will be disconnected in two hours. Press 1 and give your verification code.",
    "We detected illegal activity on your name. Transfer your savings to a safe account now.",
]
NORMAL = [
    "Hi, just checking if we are still meeting for lunch tomorrow at one.",
    "Hello, this is the clinic confirming your appointment on Thursday at ten.",
    "Can you send me the notes from yesterday's lecture when you get a chance?",
    "Hey, I will be home late tonight, please start dinner without me.",
    "Your order has been delivered. Please rate your experience when convenient.",
    "Hi, this is the library, the book you reserved is ready for pickup.",
    "Good morning, I wanted to discuss the project schedule for next week.",
    "Mom, I reached the hostel safely, I will call you after dinner.",
    "Thanks for calling, how can I help you with your internet connection today?",
    "Reminder: the team meeting has moved to three o'clock in the conference room.",
    "Hey, are you free this weekend? We are planning a small trip.",
    "Your bill for this month is available in the app, no action is needed.",
]

_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
    LogisticRegression(max_iter=1000),
).fit(SCAM + NORMAL, [1] * len(SCAM) + [0] * len(NORMAL))


def scam_score(text: str) -> float:
    """Probability (0-1) that the transcript looks like a scam call."""
    return float(_model.predict_proba([text])[0][1])


if __name__ == "__main__":
    for t in ["Share your OTP right now or your account will be blocked",
              "Are we still on for lunch tomorrow?"]:
        print(round(scam_score(t), 3), "|", t)
