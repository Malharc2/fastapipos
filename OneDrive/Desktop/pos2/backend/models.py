from pydantic import BaseModel
from bson import ObjectId
from typing import Optional, List

# Custom ObjectId handling for Pydantic
class PyObjectId(ObjectId):
    """Custom ObjectId for Pydantic."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

# User Model (Admin & Salesperson)
class UserModel(BaseModel):
    id: Optional[PyObjectId] = None
    username: str
    password: str  # Hashed password
    role: str  # 'Admin' or 'Salesperson'
    employee_code: str  # Unique employee code

# Room Management
class RoomModel(BaseModel):
    id: Optional[str] = None
    room_number: int
    assigned_to: Optional[str] = None  # Employee code of the salesperson

class RoomAssignmentModel(BaseModel):
    id: Optional[PyObjectId] = None
    room_id: str
    employee_code: str
    assigned_by: str  # Employee code of the admin

    class Config:
        json_encoders = {ObjectId: str}

# Product Category Model
class CategoryModel(BaseModel):
    id: Optional[str] = None
    category_name: str
    description: Optional[str] = None

# Product Model
class ProductModel(BaseModel):
    id: Optional[str] = None
    product_name: str
    category_id: str
    description: Optional[str] = None
    price: float = 0.0
    gst: float = 0.0
    quantity: int = 0
    image: Optional[bytes] = None  # Image of the product
    barcode: Optional[str] = None  # Barcode of the product

    def calculate_total_price_with_gst(self):
        subtotal = self.price * self.quantity
        gst_amount = subtotal * (self.gst / 100)
        return subtotal + gst_amount

# Cart Model (Adding Products to Cart)
class CartItemModel(BaseModel):
    product_id: str
    product_name: str
    barcode: Optional[str] = None
    price: float
    gst: float
    quantity: int
    total_price: float  # Price including GST

    def update_quantity(self, new_quantity: int):
        self.quantity = new_quantity
        self.total_price = (self.price * new_quantity) + ((self.price * new_quantity) * (self.gst / 100))

class CartModel(BaseModel):
    items: List[CartItemModel] = []
    total_amount: float = 0.0
    total_gst: float = 0.0
    discount_type: Optional[str] = None  # "percentage" or "fixed"
    discount_amount: Optional[float] = None

    def calculate_discounted_total(self):
        total_with_gst = self.total_amount + self.total_gst
        if self.discount_type == "percentage" and self.discount_amount:
            discount_value = total_with_gst * (self.discount_amount / 100)
            return total_with_gst - discount_value
        elif self.discount_type == "fixed" and self.discount_amount:
            return total_with_gst - self.discount_amount
        return total_with_gst

# Sales Model (Handling Transactions)
class SalesModel(BaseModel):
    id: Optional[PyObjectId] = None
    product_id: str
    quantity: int
    amount: float  # Total price after discount and GST
    profit: float
    date: str
    salesperson_employee_code: str  # Employee code of the salesperson
    assigned_room_id: str  # ID of the assigned room
    discount_type: Optional[str] = None
    discount_amount: Optional[float] = None
    payment_method: str  # "cash", "card", "gift_card"
    cash_received: Optional[float] = None
    change_given: Optional[float] = None
    gift_card_issued: Optional[float] = None  # If issued due to cash shortage

# Receipt Model
class ReceiptModel(BaseModel):
    transaction_id: str
    store_details: str
    items: List[CartItemModel]
    total_price: float
    gst_amount: float
    discount_applied: Optional[float]
    final_amount: float
    payment_method: str
    change_given: Optional[float]
    gift_card_issued: Optional[float]
    date: str

# Cash Register (Opening & Closing Balance)
class CashRegisterModel(BaseModel):
    id: Optional[PyObjectId] = None
    opening_balance: float
    cash_received: float = 0.0
    cash_given_as_change: float = 0.0
    gift_cards_issued: float = 0.0
    closing_balance: float = 0.0
    discrepancy: Optional[float] = 0.0  # If closing balance doesn't match expected

    def calculate_closing_balance(self):
        self.closing_balance = self.opening_balance + self.cash_received - self.cash_given_as_change

# Gift Card Model (Handling Cash Shortage)
class GiftCardModel(BaseModel):
    id: Optional[PyObjectId] = None
    card_number: str
    issued_value: float
    balance: float
    issued_to_customer: Optional[str] = None  # Customer identifier (optional)
    issued_by: str  # Employee code

# Sales Reports Model
class SalesReportSchema(BaseModel):
    employee_code: str
    start_date: str
    end_date: str

class AlertModel(BaseModel):
    id: Optional[PyObjectId] = None
    product_id: str
    current_quantity: int
    threshold_quantity: int
    alert_type: str  # 'low_stock', 'out_of_stock', 'expiring_soon'
    status: str  # 'pending', 'acknowledged', 'resolved'
    created_by: str  # employee_code
    created_at: str
    resolved_by: Optional[str] = None  # employee_code of admin who resolved
    resolved_at: Optional[str] = None
    notes: Optional[str] = None

class ProductReorderModel(BaseModel):
    id: Optional[PyObjectId] = None
    product_id: str
    quantity_ordered: int
    ordered_by: str  # employee_code
    order_date: str
    expected_delivery: Optional[str] = None
    status: str  # 'pending', 'delivered', 'cancelled'
    supplier_details: Optional[dict] = None
