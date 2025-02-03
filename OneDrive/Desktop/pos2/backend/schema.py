from typing import Optional, List
from pydantic import BaseModel, Field, validator
import re

# User Registration Schema
class UserRegistrationSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=6)  # 6-character length restriction
    role: str = Field(..., pattern="^(Admin|Salesperson)$", description="Role must be either 'Admin' or 'Salesperson'.")

    @validator("password")
    def validate_password(cls, value):
        """Ensure password has one uppercase, one lowercase, and is exactly 6 characters."""
        if len(value) != 6:
            raise ValueError("Password must be exactly 6 characters long.")
        if not any(char.islower() for char in value):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not any(char.isupper() for char in value):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not value.isalnum():
            raise ValueError("Password must only contain alphanumeric characters.")
        return value

# User Login Schema
class UserLoginSchema(BaseModel):
    employee_code: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=6, max_length=6)

# User Response Schema
class UserResponseSchema(BaseModel):
    username: str
    role: str
    employee_code: str

    class Config:
        min_anystr_length = 1
        anystr_strip_whitespace = True

# Room Schema
class RoomSchema(BaseModel):
    room_number: int = Field(..., ge=1)  # Ensures room numbers are positive integers

# Room Assignment Schema
class RoomAssignmentSchema(BaseModel):
    employee_code: str = Field(..., min_length=3, max_length=20)
    assigned_by: str = Field(..., min_length=3, max_length=20)

# Product Schema
class ProductSchema(BaseModel):
    product_name: str = Field(..., min_length=3, max_length=50)
    category_id: str = Field(..., min_length=10, max_length=10)
    description: Optional[str] = Field(None, min_length=0, max_length=200)
    price: float = Field(..., ge=0.0)
    gst: float = Field(..., ge=0.0)
    quantity: int = Field(..., ge=0)
    image: Optional[bytes] = None
    barcode: Optional[str] = None

# Category Schema
class CategorySchema(BaseModel):
    category_name: str = Field(..., min_length=3, max_length=50)
    description: Optional[str] = Field(None, min_length=0, max_length=200)

# Cart Item Schema
class CartItemSchema(BaseModel):
    product_id: str = Field(..., min_length=10, max_length=50)
    quantity: int = Field(..., ge=1)
    price: float = Field(..., ge=0.0)
    gst: float = Field(..., ge=0.0)
    total_price: float = Field(..., ge=0.0)

# Cart Schema
class CartSchema(BaseModel):
    items: List[CartItemSchema]
    total_amount: float = Field(..., ge=0.0)
    discount_type: Optional[str] = Field(None, pattern="^(percentage|fixed)$")
    discount_amount: Optional[float] = Field(None, ge=0.0)
    final_amount: float = Field(..., ge=0.0)

# Sales Schema
class SalesSchema(BaseModel):
    cart: CartSchema
    payment_method: str = Field(..., pattern="^(cash|card|gift_card)$")
    cash_received: Optional[float] = Field(0.0, ge=0.0)
    change_given: Optional[float] = Field(0.0, ge=0.0)
    gift_card_issued: Optional[float] = Field(0.0, ge=0.0)
    toffee_issued: Optional[float] = Field(0.0, ge=0.0)
    date: str
    employee_code: str = Field(..., min_length=3, max_length=20)
    room_id: str = Field(..., min_length=3, max_length=20)

# Cash Register Schema
class CashRegisterSchema(BaseModel):
    opening_balance: float = Field(..., ge=0.0)
    closing_balance: float = Field(..., ge=0.0)
    total_cash_sales: float = Field(..., ge=0.0)
    total_change_given: float = Field(..., ge=0.0)
    total_gift_cards_issued: float = Field(..., ge=0.0)
    total_toffees_issued: float = Field(..., ge=0.0)
    discrepancy: Optional[float] = Field(0.0)
    date: str

# Sales Report Schema
class SalesReportSchema(BaseModel):
    start_date: str
    end_date: str
    employee_code: Optional[str] = None
    product_id: Optional[str] = None
    room_id: Optional[str] = None
    total_sales: Optional[float] = Field(0.0, ge=0.0)
    total_profit: Optional[float] = Field(0.0, ge=0.0)
    total_discount: Optional[float] = Field(0.0, ge=0.0)
    total_cash_sales: Optional[float] = Field(0.0, ge=0.0)
    total_gift_cards_used: Optional[float] = Field(0.0, ge=0.0)
    total_toffees_issued: Optional[float] = Field(0.0, ge=0.0)
    opening_balance: Optional[float] = Field(0.0, ge=0.0)
    closing_balance: Optional[float] = Field(0.0, ge=0.0)
    discrepancy: Optional[float] = Field(0.0, ge=0.0)
class AlertSchema(BaseModel):
    product_id: str = Field(..., min_length=10, max_length=50)
    threshold_quantity: int = Field(..., ge=1)
    alert_type: str = Field(..., pattern="^(low_stock|out_of_stock|expiring_soon)$")
    notes: Optional[str] = None

class ProductReorderSchema(BaseModel):
    product_id: str = Field(..., min_length=10, max_length=50)
    quantity_ordered: int = Field(..., ge=1)
    expected_delivery: Optional[str] = None
    supplier_details: Optional[dict] = None
