from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, IndexModel

# MongoDB Connection Setup
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["company"]

# Collection Definitions
users_collection = db["users"]
rooms_collection = db["rooms"]
room_assignments_collection = db["room_assignments"]
categories_collection = db["categories"]
products_collection = db["products"]
sales_collection = db["sales"]
cart_items_collection = db["cart_items"]
carts_collection = db["carts"]
receipts_collection = db["receipts"]
cash_registers_collection = db["cash_registers"]
gift_cards_collection = db["gift_cards"]
sales_reports_collection = db["sales_reports"]
# New collections for alert system
alerts_collection = db["alerts"]
product_reorders_collection = db["product_reorders"]

async def create_indexes():
    """Create indexes for fast querying and ensure uniqueness where required."""
    
    # User collection
    await users_collection.create_index("employee_code", unique=True)
    await users_collection.create_index("email", unique=True, sparse=True)  # Added email index
    
    # Room collections
    await rooms_collection.create_index("room_number", unique=True)
    await room_assignments_collection.create_index([("room_id", ASCENDING), ("date", ASCENDING)], unique=True)
    
    # Product & Category collections
    await categories_collection.create_index("category_name", unique=True)
    await products_collection.create_index("id", unique=True)
    await products_collection.create_index("barcode", unique=True)
    await products_collection.create_index("category_id")  # Added for category lookups
    
    # Sales collections
    await sales_collection.create_index("id", unique=True)
    await sales_collection.create_index([
        ("date", ASCENDING),
        ("employee_code", ASCENDING),
        ("product_id", ASCENDING)
    ])
    
    # Cart collections
    await carts_collection.create_index("id", unique=True)
    await carts_collection.create_index([("user_id", ASCENDING), ("status", ASCENDING)])
    await cart_items_collection.create_index([("cart_id", ASCENDING), ("product_id", ASCENDING)])
    
    # Receipts
    await receipts_collection.create_index("id", unique=True)
    await receipts_collection.create_index("sale_id")
    await receipts_collection.create_index("date")
    
    # Cash register collections
    await cash_registers_collection.create_index([("date", ASCENDING), ("register_id", ASCENDING)], unique=True)
    
    # Gift cards
    await gift_cards_collection.create_index("id", unique=True)
    await gift_cards_collection.create_index("assigned_to")
    await gift_cards_collection.create_index([("status", ASCENDING), ("expiry_date", ASCENDING)])
    
    # Sales Reports
    await sales_reports_collection.create_index([
        ("start_date", ASCENDING),
        ("end_date", ASCENDING),
        ("employee_code", ASCENDING),
        ("room_id", ASCENDING)
    ])
    
    # New indexes for Alert System
    # Alerts collection
    await alerts_collection.create_index("product_id")
    await alerts_collection.create_index([("status", ASCENDING)])
    await alerts_collection.create_index([
        ("product_id", ASCENDING),
        ("status", ASCENDING),
        ("created_at", DESCENDING)
    ])
    await alerts_collection.create_index([
        ("alert_type", ASCENDING),
        ("created_at", DESCENDING)
    ])
    
    # Product Reorders collection
    await product_reorders_collection.create_index("product_id")
    await product_reorders_collection.create_index([("status", ASCENDING)])
    await product_reorders_collection.create_index([
        ("product_id", ASCENDING),
        ("order_date", DESCENDING)
    ])
    await product_reorders_collection.create_index([
        ("ordered_by", ASCENDING),
        ("order_date", DESCENDING)
    ])