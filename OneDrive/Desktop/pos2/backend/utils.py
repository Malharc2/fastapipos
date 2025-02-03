import random
import string
import uuid
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        return False

def generate_employee_code(role: str) -> str:
    prefix = "ADM" if role == "Admin" else "SAL"
    suffix = ''.join(random.choices(string.digits, k=5))
    return f"{prefix}-{suffix}"

def generate_category_id() -> str:
    return f"CAT-{random.randint(100000, 999999):06x}"

def generate_product_id() -> str:
    return f"PRO-{random.randint(100000, 999999):06x}"

def generate_barcode():
    return f"BAR-{uuid.uuid4().hex[:6].upper()}"

def generate_transaction_id() -> str:
    """Generate a unique transaction ID."""
    return f"TRN-{uuid.uuid4().hex[:8].upper()}"

def generate_gift_card_id() -> str:
    """Generate a unique gift card ID."""
    return f"GFT-{uuid.uuid4().hex[:8].upper()}"
