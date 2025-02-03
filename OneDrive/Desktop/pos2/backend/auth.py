import base64
import io
import json
import os
from typing import List, Optional
import uuid
import aiohttp
from fastapi import APIRouter, Body, File, Form, HTTPException, Depends, UploadFile, requests
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import pandas as pd
import pyzbar
import tabula
from models import UserModel, RoomModel, RoomAssignmentModel, CategoryModel, ProductModel, SalesModel
from schema import SalesReportSchema, SalesSchema, UserRegistrationSchema, UserLoginSchema, UserResponseSchema, RoomSchema, RoomAssignmentSchema, CategorySchema, ProductSchema
from database import db, users_collection, rooms_collection, room_assignments_collection, categories_collection, products_collection,sales_collection
from utils import generate_gift_card_id, generate_transaction_id, hash_password, verify_password, generate_employee_code, generate_category_id, generate_product_id, generate_barcode
from bson import Binary, ObjectId
from fpdf import FPDF
import zipfile
import tempfile
import datetime
from barcode import Code128
from barcode.writer import ImageWriter
auth_router = APIRouter()

@auth_router.post("/register", response_model=dict)
async def register_user(user: UserRegistrationSchema):
    # Check if the user already exists
    user_exists = await db.users.find_one({"username": user.username})
    if user_exists:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Generate a unique employee code
    employee_code = generate_employee_code(user.role)

    # Hash the password
    hashed_password = hash_password(user.password)

    # Create the new user
    new_user = {
        "username": user.username,
        "password": hashed_password,
        "role": user.role,
        "employee_code": employee_code
    }

    result = await db.users.insert_one(new_user)
    user_data = await db.users.find_one({"username": user.username}, {"_id": 0, "password": 0})

    return {
        "success": True,
        "status_code": 200,
        "message": f"Welcome, {user_data['role']} with employee code {user_data['employee_code']}",
        "data": user_data
    }

@auth_router.post("/login")
async def login(user: UserLoginSchema):
    # Check if the user exists based on employee_code
    user_data = await users_collection.find_one({"employee_code": user.employee_code})

    # If user with the employee_code doesn't exist
    if user_data is None:
        raise HTTPException(status_code=400, detail="Invalid employee code")

    # Verify the password
    if not verify_password(user.password, user_data["password"]):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Return a success message with role (Admin/Salesperson) after successful authentication
    return {"success": True, "status_code": 200, "message": f"Welcome, {user_data['role']} with employee code {user_data['employee_code']}", "data": jsonable_encoder(user_data, exclude={"_id", "password"})}

@auth_router.post("/reset-employee-code")
async def reset_employee_code(username: str, role: str):
    user_data = await users_collection.find_one({"username": username, "role": role})
    if user_data is None:
        raise HTTPException(status_code=400, detail="Invalid username or role")
    
    new_employee_code = generate_employee_code(role)
    await users_collection.update_one({"username": username, "role": role}, {"$set": {"employee_code": new_employee_code}})
    
    return {"success": True, "status_code": 200, "message": "Employee code reset successfully", "data": {"employee_code": new_employee_code}}

@auth_router.post("/validate-empcode")
async def validate_employee_code(emp_code: str, role: str):
    user = await db.users.find_one({"employee_code": emp_code, "role": role})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")
    return {"success": True, "status_code": 200, "message": f"Access granted for {role}."}
@auth_router.get("/get-all-users")
async def get_all_users():
    # Get all users
    all_users = await users_collection.find().to_list(None)

    # Convert ObjectId to string
    for u in all_users:
        if "_id" in u:
            u["_id"] = str(u["_id"])
        if "password" in u:
            del u["password"]

    return {"success": True, "status_code": 200, "message": "Users retrieved successfully", "data": all_users}
@auth_router.get("/get-user-details-by-employee-id")
async def get_user_details_by_employee_id(employee_id: str):
    # Get the user by employee ID
    user = await users_collection.find_one({"employee_code": employee_id})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Convert ObjectId to string
    if "_id" in user:
        user["_id"] = str(user["_id"])
    if "password" in user:
        del user["password"]

    return {"success": True, "status_code": 200, "message": "User details retrieved successfully", "data": user}

@auth_router.post("/assign-room")
async def assign_room(room_assignment: RoomAssignmentSchema):
    # Check if the admin is assigning the room
    admin = await users_collection.find_one({"employee_code": room_assignment.assigned_by, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Only admins can assign rooms")

    # Check if the salesperson exists
    salesperson = await users_collection.find_one({"employee_code": room_assignment.employee_code, "role": "Salesperson"})
    if not salesperson:
        raise HTTPException(status_code=404, detail="Salesperson not found")

    # Check if a room has already been assigned to the salesperson
    existing_assignment = await room_assignments_collection.find_one({"employee_code": room_assignment.employee_code})
    if existing_assignment:
        raise HTTPException(status_code=400, detail=f"Room already assigned to {salesperson['username']}. Please add another employee code.")

    # Generate a unique room ID
    room_id = await generate_room_id_from_db()

    # Assign the room to the salesperson
    await room_assignments_collection.insert_one({
        "room_id": room_id,
        "employee_code": room_assignment.employee_code,
        "assigned_by": room_assignment.assigned_by
    })

    return {"success": True, "status_code": 200, "message": f"Room {room_id} assigned to {salesperson['username']}"}

async def generate_room_id_from_db():
    # Get the last room ID from the database
    last_room = await rooms_collection.find_one(sort=[("id", -1)])
    if last_room:
        last_room_id = int(last_room["id"][2:])  # Remove the 'RO' prefix
        new_room_id = f"RO{last_room_id + 1}"
    else:
        new_room_id = "RO1000"

    # Check if the new room ID already exists
    while await rooms_collection.find_one({"id": new_room_id}):
        last_room_id = int(new_room_id[2:])  # Remove the 'RO' prefix
        new_room_id = f"RO{last_room_id + 1}"

    # Create the new room
    await rooms_collection.insert_one({
        "id": new_room_id,
        "room_number": int(new_room_id[2:])  # Remove the 'RO' prefix
    })

    return new_room_id

@auth_router.get("/get-room-details")
async def get_room_details(employee_code: str):
    assigned_rooms = await room_assignments_collection.find({"employee_code": employee_code}).to_list(None)
    
    # Convert ObjectId to string
    for room in assigned_rooms:
        if "_id" in room:
            room["_id"] = str(room["_id"])
    
    rooms = await rooms_collection.find().to_list(None)
    rooms_dict = [jsonable_encoder(room, exclude={"_id"}) for room in rooms]
    
    # Check if the room is assigned to the salesperson
    assigned_room_ids = [room["room_id"] for room in assigned_rooms]
    for room in rooms_dict:
        if room["id"] in assigned_room_ids:
            room["assigned"] = True
            room["assigned_to"] = employee_code
        else:
            room["assigned"] = False
            room["assigned_to"] = None
    
    if not assigned_rooms:
        return {"success": False, "status_code": 404, "message": "No rooms assigned to this employee"}
    else:
        filtered_rooms = [room for room in rooms_dict if room["assigned"]]
        if not filtered_rooms:
            return {"success": False, "status_code": 404, "message": "No rooms assigned to this employee"}
        else:
            return {"success": True, "status_code": 200, "message": "Room details retrieved successfully", "data": filtered_rooms}

@auth_router.post("/create-category")
async def create_category(category: CategorySchema, employee_code: str):
    # Check if the admin exists in the database
    admin = await users_collection.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Generate a unique category ID
    category_id = generate_category_id()

    # Create the category
    await categories_collection.insert_one({
        "id": category_id,
        "category_name": category.category_name,
        "description": category.description
    })

    return {"success": True, "status_code": 200, "message": f"Category {category.category_name} created with ID {category_id}"}

@auth_router.post("/update-category")
async def update_category(category_id: str, category: CategorySchema, employee_code: str):
    # Check if the admin exists in the database
    admin = await users_collection.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Check if the category exists
    category_data = await categories_collection.find_one({"id": category_id})
    if not category_data:
        raise HTTPException(status_code=404, detail="Category not found")

    # Update the category
    await categories_collection.update_one({"id": category_id}, {"$set": {
        "category_name": category.category_name,
        "description": category.description
    }})

    return {"success": True, "status_code": 200, "message": f"Category {category.category_name} updated successfully"}

import base64

@auth_router.post("/add-product")
async def add_product(
    employee_code: str,
    product_name: str = Form(...),
    category_id: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    gst: float = Form(...),
    quantity: int = Form(...),
    barcode: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Check if the category exists
    category = await categories_collection.find_one({"id": category_id})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Prepare product data
    product_data = {
        "id": generate_product_id(),
        "product_name": product_name,
        "category_id": category_id,
        "description": description,
        "price": price,
        "gst": gst,
        "quantity": quantity,
        "barcode": barcode,
        "image": None,  # Default to None
    }

    # Convert image bytes to Base64 string if provided
    if image:
        image_bytes = await image.read()
        product_data["image"] = base64.b64encode(image_bytes).decode("utf-8")

    # Validate and insert into DB
    try:
        product_model = ProductModel(**product_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    await products_collection.insert_one(product_model.dict())

    return {
        "success": True,
        "status_code": 200,
        "message": "Product added successfully",
        "data": product_model.dict(),
    }


@auth_router.get("/get-products")
async def get_products():
    products = []
    cursor = products_collection.find({})
    async for product in cursor:
        product_data = dict(product)
        
        # Remove MongoDB _id
        if "_id" in product_data:
            del product_data["_id"]
            
        # Convert any bytes fields to strings
        for key, value in product_data.items():
            if isinstance(value, bytes):
                try:
                    product_data[key] = value.decode('utf-8')
                except UnicodeDecodeError:
                    # If decoding fails, convert to base64 string
                    product_data[key] = base64.b64encode(value).decode('utf-8')
            
        products.append(product_data)
    
    return JSONResponse(content={"success": True, "status_code": 200, "data": products})

@auth_router.put("/update-product")
async def update_product(
    employee_code: str,
    product_id: str = Form(...),
    product_name: str = Form(...),
    category_id: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    gst: float = Form(...),
    quantity: int = Form(...),
    barcode: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Check if the product exists
    product = await products_collection.find_one({"id": product_id})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Check if the category exists
    category = await categories_collection.find_one({"id": category_id})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Update the product
    product_data = {
        "id": product_id,  # Include the id field
        "product_name": product_name,
        "category_id": category_id,
        "description": description,
        "price": price,
        "gst": gst,
        "quantity": quantity,
        "barcode": barcode,
    }

    if image:
        product_data["image"] = await image.read()

    try:
        product_model = ProductModel(**product_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert the product model to a dictionary
    product_dict = product_model.dict()

    # If the image is present, convert it to a binary object
    if "image" in product_dict:
        product_dict["image"] = Binary(product_dict["image"])

    # Update the product in the database
    await products_collection.update_one({"id": product_id}, {"$set": product_dict})

    return {"success": True, "status_code": 200, "message": "Product updated successfully"}

@auth_router.get("/get-categories")
async def get_categories(employee_code: str):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    categories = await categories_collection.find().to_list(None)
    categories_dict = [jsonable_encoder(category, exclude={"_id"}) for category in categories]
    return {"success": True, "status_code": 200, "message": "Categories retrieved successfully", "data": categories_dict}

@auth_router.get("/get-products")
async def get_products(employee_code: str):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    products = await products_collection.find().to_list(None)
    products_dict = []
    for product in products:
        product_data = product.copy()
        if "_id" in product_data:
            product_data["_id"] = str(product_data["_id"])
        if "image" in product_data and product_data["image"] is not None:
            # Encode the binary image data as a base64 encoded string
            product_data["image"] = base64.b64encode(product_data["image"]).decode("utf-8")
        products_dict.append(product_data)

    return {"success": True, "status_code": 200, "message": "Products retrieved successfully", "data": products_dict}
@auth_router.get("/get-products-by-category")
async def get_products_by_category(employee_code: str, category_id: str):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    products = await products_collection.find({"category_id": category_id}).to_list(None)
    products_dict = [jsonable_encoder(product, exclude={"_id", "image"}) for product in products]
    return {"success": True, "status_code": 200, "message": "Products retrieved successfully", "data": products_dict}
@auth_router.put("/update-product-by-id")
async def update_product_by_id(
    employee_code: str,
    product_id: str = Form(...),
    product_name: str = Form(...),
    category_id: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    gst: float = Form(...),
    quantity: int = Form(...),
    barcode: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Check if the product exists
    product = await products_collection.find_one({"id": product_id})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Check if the category exists
    category = await categories_collection.find_one({"id": category_id})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Update the product
    product_data = {
        "id": product_id,  # Include the id field
        "product_name": product_name,
        "category_id": category_id,
        "description": description,
        "price": price,
        "gst": gst,
        "quantity": quantity,
        "barcode": barcode,
    }

    if image:
        product_data["image"] = await image.read()

    try:
        product_model = ProductModel(**product_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert the product model to a dictionary
    product_dict = product_model.dict()

    # If the image is present, convert it to a binary object
    if "image" in product_dict:
        product_dict["image"] = Binary(product_dict["image"])

    # Update the product in the database
    await products_collection.update_one({"id": product_id}, {"$set": product_dict})

    return {"success": True, "status_code": 200, "message": "Product updated successfully"}

@auth_router.get("/get-product-by-id")
async def get_product_by_id(product_id: str):
    # Get the product by ID
    product = await products_collection.find_one({"id": product_id})

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Convert ObjectId to string
    if "_id" in product:
        product["_id"] = str(product["_id"])

    # Convert any bytes fields to strings
    for key, value in product.items():
        if isinstance(value, bytes):
            try:
                product[key] = value.decode('utf-8')
            except UnicodeDecodeError:
                # If decoding fails, convert to base64 string
                product[key] = base64.b64encode(value).decode('utf-8')

    return {"success": True, "status_code": 200, "message": "Product retrieved successfully", "data": product}

@auth_router.post("/add-multiple-products")
async def add_multiple_products(employee_code: str, file: UploadFile = File(...)):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Check if the file is in the correct format
    if file.filename.split('.')[-1].lower() not in ['xlsx', 'xls', 'csv', 'pdf']:
        raise HTTPException(status_code=400, detail="Invalid file format. Only Excel, CSV, and PDF files are supported.")

    # Read the file
    try:
        if file.filename.split('.')[-1].lower() in ['xlsx', 'xls']:
            df = pd.read_excel(file.file)
        elif file.filename.split('.')[-1].lower() == 'csv':
            df = pd.read_csv(file.file)
        elif file.filename.split('.')[-1].lower() == 'pdf':
            import tabula
            df = tabula.read_pdf(file.file, pages='all')[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid file format.")

    errors = []
    successful_products = 0

    # Add the products to the database
    for index, row in df.iterrows():
        try:
            # Handle image downloading and processing
            image_data = None
            if "image_url" in row and not pd.isna(row["image_url"]):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(str(row["image_url"])) as response:
                            if response.status == 200:
                                image_bytes = await response.read()
                                # Convert to base64
                                image_data = base64.b64encode(image_bytes).decode('utf-8')
                except Exception as e:
                    errors.append(f"Error downloading image for product {row['product_name']}: {str(e)}")

            product = {
                "id": generate_product_id(),
                "product_name": str(row["product_name"]),
                "category_id": str(row["category_id"]),
                "description": str(row["description"]) if not pd.isna(row["description"]) else None,
                "price": float(row["price"]),
                "gst": float(row["gst"]),
                "discount": float(row["discount"]) if not pd.isna(row["discount"]) else None,
                "discount_amount": float(row["discount_amount"]) if not pd.isna(row["discount_amount"]) else None,
                "quantity": 0 if pd.isna(row["quantity"]) else int(row["quantity"]),
                "barcode": str(row["barcode"]) if not pd.isna(row["barcode"]) else None,
                "image": image_data
            }

            product_model = ProductModel(**product)
            await products_collection.insert_one(jsonable_encoder(product_model))
            successful_products += 1
        
        except Exception as e:
            errors.append(f"Error processing product {index + 1}: {str(e)}")
            continue

    # Prepare response
    response = {
        "success": len(errors) == 0,
        "status_code": 200 if len(errors) == 0 else 400,
        "message": f"Successfully added {successful_products} products" + 
                    (f" with {len(errors)} errors" if errors else ""),
        "total_processed": len(df),
        "successful_products": successful_products
    }

    if errors:
        response["errors"] = errors

    return response

# Endpoint to add products to cart by barcode scanning
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
@auth_router.post("/add-to-cart-by-barcode")
async def add_to_cart_by_barcode(barcode: str, quantity: int, employee_code: str, room_id: str):
    # Fetch product details from the database
    product = await products_collection.find_one({"barcode": barcode})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Check if the product is already in the cart
    cart = await carts_collection.find_one({
        "product_id": product["id"], 
        "employee_code": employee_code, 
        "room_id": room_id
    })
    
    if cart:
        # Increase the quantity if the product is already in the cart
        await carts_collection.update_one(
            {"product_id": product["id"], "employee_code": employee_code, "room_id": room_id},
            {"$inc": {"quantity": quantity}}
        )
        # Fetch the updated cart
        cart = await carts_collection.find_one({
            "product_id": product["id"],
            "employee_code": employee_code,
            "room_id": room_id
        })
    else:
        # Fetch the category information
        category = await categories_collection.find_one({"id": product["category_id"]})
        
        # Handle case where category is not found
        category_name = "Unknown Category"
        if category:
            category = jsonable_encoder(category)
            category_name = category.get("category_name", "Unknown Category")
        
        # Handle image conversion
        product_image = None
        if "image" in product and product["image"] is not None:
            if isinstance(product["image"], bytes):
                product_image = base64.b64encode(product["image"]).decode("utf-8")
            else:
                product_image = product["image"]
        
        # Create new cart entry
        new_cart = {
            "product_id": str(product["id"]),
            "product_name": product["product_name"],
            "price": product["price"],
            "gst": product["gst"],
            "quantity": quantity,
            "total_price": product["price"] * quantity,
            "gst_amount": (product["price"] * quantity) * (product["gst"] / 100),
            "image": product_image,
            "category_id": str(product["category_id"]),
            "category_name": category_name,
            "employee_code": employee_code,
            "room_id": room_id
        }
        
        await carts_collection.insert_one(new_cart)
        cart = new_cart
    
    # Convert ObjectId to string if present
    if "_id" in cart:
        cart["_id"] = str(cart["_id"])
    
    # Convert any remaining bytes to base64
    for key, value in cart.items():
        if isinstance(value, bytes):
            cart[key] = base64.b64encode(value).decode("utf-8")
    
    return {
        "success": True,
        "status_code": 200,
        "message": "Product added to cart successfully",
        "data": cart
    }

# Endpoint to add products to cart by manual barcode entry
@auth_router.post("/add-to-cart")
async def add_to_cart(product_id: str, quantity: int, employee_code: str, room_id: str):
    # Validate room assignment for the employee
    room_assignment = await room_assignments_collection.find_one({
        "employee_code": employee_code,
        "room_id": room_id
    })
    if not room_assignment:
        raise HTTPException(
            status_code=403,
            detail="Employee is not assigned to this room."
        )
    
    # Fetch product details from the database
    product = await products_collection.find_one({"id": product_id})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Fetch category details and handle potential missing category
    category = await categories_collection.find_one({"id": product["category_id"]})
    if not category:
        raise HTTPException(
            status_code=404, 
            detail=f"Category not found for product {product_id}"
        )
    
    # Check if the product is already in the cart
    cart = await carts_collection.find_one({
        "product_id": product_id, 
        "employee_code": employee_code, 
        "room_id": room_id
    })
    
    if cart:
        # Increase the quantity if the product is already in the cart
        await carts_collection.update_one(
            {
                "product_id": product_id, 
                "employee_code": employee_code, 
                "room_id": room_id
            },
            {"$inc": {"quantity": quantity}}
        )
    else:
        # Add the product to the cart if it's not already there
        await carts_collection.insert_one({
            "product_id": product_id,
            "product_name": product["product_name"],
            "price": product["price"],
            "gst": product["gst"],
            "quantity": quantity,
            "image": product["image"],
            "category_id": product["category_id"],
            "category_name": category["category_name"],
            "employee_code": employee_code,
            "room_id": room_id
        })
    
    return {
        "success": True,
        "status_code": 200,
        "message": "Product added to cart successfully"
    }


# Endpoint to get cart details
@auth_router.get("/get-total-price-after-adding-to-cart")
async def get_total_price_after_adding_to_cart(employee_code: str, room_id: str):
    # Fetch cart details from the database
    cart = await carts_collection.find({"employee_code": employee_code, "room_id": room_id}).to_list(None)
    
    if not cart:
        raise HTTPException(status_code=404, detail="Cart is empty")
    
    # Calculate the total price
    total_price = 0
    gst_amount = 0
    for item in cart:
        total_price += item["price"] * item["quantity"]
        gst_amount += (item["price"] * item["quantity"]) * (item["gst"] / 100)
    
    return {
        "success": True,
        "status_code": 200,
        "message": "Total price retrieved successfully",
        "data": {
            "total_price": total_price,
            "gst_amount": gst_amount,
            "total_amount": total_price + gst_amount
        }
    }
@auth_router.get("/get-cart")
async def get_cart(employee_code: str, room_id: str):
    # Fetch cart details from the database
    cart = await carts_collection.find({
        "employee_code": employee_code, 
        "room_id": room_id
    }).to_list(None)
    
    # Convert ObjectId to string and handle image encoding
    for item in cart:
        # Convert ObjectId to string
        if "_id" in item:
            item["_id"] = str(item["_id"])
        
        # Handle image
        if "image" in item and item["image"] is not None:
            # If image is already a string (likely already base64), keep it as is
            if isinstance(item["image"], str):
                continue
            # If image is bytes, encode it
            elif isinstance(item["image"], bytes):
                item["image"] = base64.b64encode(item["image"]).decode("utf-8")
            # If image is None or any other type, set to None
            else:
                item["image"] = None
    
    return {
        "success": True,
        "status_code": 200,
        "message": "Cart retrieved successfully",
        "data": cart
    }
# Endpoint to update cart quantity
@auth_router.put("/update-cart-quantity")
async def update_cart_quantity(product_id: str, quantity: int, employee_code: str, room_id: str):
    # Fetch cart details from the database
    cart = await carts_collection.find_one({"product_id": product_id, "employee_code": employee_code, "room_id": room_id})
    
    if not cart:
        raise HTTPException(status_code=404, detail="Product not found in cart")
    
    # Calculate the total price
    total_price = cart["price"] * quantity
    
    # Calculate the GST
    gst = total_price * (cart["gst"] / 100)
    
    # Update the cart quantity
    await carts_collection.update_one({"product_id": product_id, "employee_code": employee_code, "room_id": room_id}, {"$set": {
        "quantity": quantity,
        "total_price": total_price,
        "gst_amount": gst
    }})
    
    return {"success": True, "status_code": 200, "message": "Cart quantity updated successfully"}

# Endpoint to remove product from cart
@auth_router.delete("/remove-from-cart")
async def remove_from_cart(product_id: str, employee_code: str, room_id: str):
    # Fetch cart details from the database
    cart = await carts_collection.find_one({"product_id": product_id, "employee_code": employee_code, "room_id": room_id})
    
    if not cart:
        raise HTTPException(status_code=404, detail="Product not found in cart")
    
    # Remove the product from the cart
    await carts_collection.delete_one({"product_id": product_id, "employee_code": employee_code, "room_id": room_id})
    
    return {"success": True, "status_code": 200, "message": "Product removed from cart successfully"}

# Endpoint to apply discount
@auth_router.post("/apply-discount-to-product")
async def apply_discount_to_product(
    product_id: str, 
    discount_type: str, 
    discount_amount: float, 
    employee_code: str, 
    room_id: str
):
    # Fetch cart details from the database
    cart = await carts_collection.find_one({"employee_code": employee_code, "room_id": room_id, "product_id": product_id})
    
    if not cart:
        raise HTTPException(status_code=404, detail="Product not found in cart")
    
    # Calculate the total price
    total_price = cart["price"] * cart["quantity"]
    
    # Calculate the GST
    gst = total_price * (cart["gst"] / 100)
    
    # Apply the discount
    if discount_type == "percentage":
        discount_amount = total_price * (discount_amount / 100)
    elif discount_type == "fixed":
        discount_amount = discount_amount
    
    new_total = total_price - discount_amount
    
    # Update the cart with the new total
    await carts_collection.update_one({"employee_code": employee_code, "room_id": room_id, "product_id": product_id}, {"$set": {
        "price": new_total,
        "gst": gst,
        "discount_type": discount_type,
        "discount_amount": discount_amount
    }})
    
    return {"success": True, "status_code": 200, "message": "Discount applied successfully"}

@auth_router.get("/get-product-total-price-after-discount")
async def get_product_total_price_after_discount(
    product_id: str, 
    employee_code: str, 
    room_id: str
):
    # Fetch cart details from the database
    cart = await carts_collection.find_one({"employee_code": employee_code, "room_id": room_id, "product_id": product_id})
    
    if not cart:
        raise HTTPException(status_code=404, detail="Product not found in cart")
    
    # Calculate the total price
    total_price = cart["price"] * cart["quantity"]
    
    # Calculate the GST
    gst = total_price * (cart["gst"] / 100)
    
    # Return the total price after discount
    return {
        "success": True,
        "status_code": 200,
        "message": "Total price after discount retrieved successfully",
        "data": {
            "product_name": cart["product_name"],
            "price": cart["price"],
            "gst": cart["gst"],
            "quantity": cart["quantity"],
            "total_price": total_price,
            "discount_type": cart.get("discount_type", None),
            "discount_amount": cart.get("discount_amount", None),
            "gst_amount": gst,
            "total_amount": total_price + gst
        }
    }


# Endpoint to process payment
@auth_router.post("/process-payment")
async def process_payment(payment_method: str, amount: float, employee_code: str, room_id: str):
    global cash_register_balance
    
    # Fetch cart details from the database
    cart = await carts_collection.find({"employee_code": employee_code, "room_id": room_id}).to_list(None)
    
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found")
    
    # Calculate the total price
    total_price = sum(item["total_price"] for item in cart)
    
    # Calculate the GST
    gst = sum(item["gst_amount"] for item in cart)
    
    # Process the payment
    if payment_method == "cash":
        # Calculate the change
        change = amount - total_price
        
        # Update the cash register balance
        cash_register_balance += amount
        
        # Clear the cart
        await carts_collection.delete_many({"employee_code": employee_code, "room_id": room_id})
        
        # Generate a receipt
        receipt = await generate_receipt(cart, employee_code, room_id, total_price, gst, change)
        
        return {"success": True, "status_code": 200, "message": "Payment processed successfully", "change": change, "receipt": receipt}
    elif payment_method == "phonepay":
        # Integrate with PhonePay payment gateway
        url = "https://api.phonepe.com/v1/merchant/transactions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_PHONEPAY_API_KEY"
        }
        data = {
            "merchant_id": "YOUR_MERCHANT_ID",
            "transaction_id": "YOUR_TRANSACTION_ID",
            "amount": total_price,
            "currency": "INR"
        }
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            # Update the cash register balance
            cash_register_balance += total_price
            
            # Clear the cart
            await carts_collection.delete_many({"employee_code": employee_code, "room_id": room_id})
            
            # Generate a receipt
            receipt = await generate_receipt(cart, employee_code, room_id, total_price, gst, 0)
            
            return {"success": True, "status_code": 200, "message": "Payment processed successfully", "receipt": receipt}
        else:
            raise HTTPException(status_code=400, detail="Payment failed")

async def generate_receipt(cart, employee_code, room_id, total_price, gst, change):
    # Get the salesperson details
    salesperson = await users_collection.find_one({"employee_code": employee_code})
    
    # Get the assigned room details
    room_assignment = await room_assignments_collection.find_one({"employee_code": employee_code})
    room = await rooms_collection.find_one({"id": room_assignment["room_id"]})
    
    # Create a receipt PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(200, 10, txt="Receipt", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Salesperson: {salesperson['username']}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"Employee Code: {salesperson['employee_code']}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"Room Number: {room['room_number']}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}", ln=True, align='L')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="Product Details:", ln=True, align='L')
    for item in cart:
        pdf.cell(0, 10, txt=f"{item['product_name']}: {item['quantity']} x {item['price']}", ln=True, align='L')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Subtotal: {total_price - gst}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"GST: {gst}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"Total: {total_price}", ln=True, align='L')
    pdf.cell(0, 10, txt=f"Change: {change}", ln=True, align='L')
    
    # Save the receipt to a file
    receipt_file = f"receipt_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(receipt_file)
    
    return receipt_file

# Endpoint to get opening and closing balance
@auth_router.get("/get-opening-and-closing-balance")
async def get_opening_and_closing_balance(employee_code: str, room_id: str):
    global cash_register_balance
    
    # Get the opening balance
    opening_balance = cash_register_balance
    
    # Get the closing balance
    closing_balance = cash_register_balance
    
    # Return the opening and closing balance
    return {"success": True, "status_code": 200, "message": "Opening and closing balance retrieved successfully", "opening_balance": opening_balance, "closing_balance": closing_balance}

# Endpoint to get daily sales reports
@auth_router.get("/get-daily-sales-reports")
async def get_daily_sales_reports(employee_code: str, room_id: str):
    # Check if the admin exists in the database
    admin = await users_collection.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")
    
    # Fetch sales data from the database
    sales_data = await sales_collection.find().to_list(None)
    
    # Calculate the total sales
    total_sales = 0
    for sale in sales_data:
        total_sales += sale["amount"]
    
    # Calculate the total discounts applied
    total_discounts = 0
    for sale in sales_data:
        total_discounts += sale["discount_amount"]
    
    # Calculate the number of transactions
    num_transactions = len(sales_data)
    
    # Calculate the gift cards issued and used
    gift_cards_issued = 0
    gift_cards_used = 0
    for sale in sales_data:
        if sale["gift_card_id"]:
            gift_cards_issued += 1
            gift_cards_used += 1
    
    # Calculate the discrepancy
    discrepancy = 0
    
    # Return the daily sales reports
    return {"success": True, "status_code": 200, "message": "Daily sales reports retrieved successfully", "total_sales": total_sales, "total_discounts": total_discounts, "num_transactions": num_transactions, "gift_cards_issued": gift_cards_issued, "gift_cards_used": gift_cards_used, "discrepancy": discrepancy}

# Endpoint to get monthly sales reports
@auth_router.post("/get-monthly-sales-reports")
async def get_monthly_sales_reports(employee_code: str, room_id: str, month: int, year: int):
    # Check if the admin exists in the database
    admin = await users_collection.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")
    
    # Fetch sales data from the database
    sales_data = await sales_collection.find({"date": {"$regex": f"^{year}-{month}-"}}).to_list(None)
    
    # Calculate the total sales
    total_sales = 0
    for sale in sales_data:
        total_sales += sale["amount"]
    
    # Calculate the total discounts applied
    total_discounts = 0
    for sale in sales_data:
        total_discounts += sale["discount_amount"]
    
    # Calculate the number of transactions
    num_transactions = len(sales_data)
    
    # Calculate the gift cards issued and used
    gift_cards_issued = 0
    gift_cards_used = 0
    for sale in sales_data:
        if sale["gift_card_id"]:
            gift_cards_issued += 1
            gift_cards_used += 1
    
    # Calculate the discrepancy
    discrepancy = 0
    
    # Return the monthly sales reports
    return {"success": True, "status_code": 200, "message": "Monthly sales reports retrieved successfully", "total_sales": total_sales, "total_discounts": total_discounts, "num_transactions": num_transactions, "gift_cards_issued": gift_cards_issued, "gift_cards_used": gift_cards_used, "discrepancy": discrepancy}

# Endpoint to get yearly sales reports
@auth_router.post("/get-yearly-sales-reports")
async def get_yearly_sales_reports(employee_code: str, room_id: str, year: int):
    # Check if the admin exists in the database
    admin = await users_collection.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")
    
    # Fetch sales data from the database
    sales_data = await sales_collection.find({"date": {"$regex": f"^{year}-"}}).to_list(None)
    
    # Calculate the total sales
    total_sales = 0
    for sale in sales_data:
        total_sales += sale["amount"]
    
    # Calculate the total discounts applied
    total_discounts = 0
    for sale in sales_data:
        total_discounts += sale["discount_amount"]
    
    # Calculate the number of transactions
    num_transactions = len(sales_data)
    
    # Calculate the gift cards issued and used
    gift_cards_issued = 0
    gift_cards_used = 0
    for sale in sales_data:
        if sale["gift_card_id"]:
            gift_cards_issued += 1
            gift_cards_used += 1
    
    # Calculate the discrepancy
    discrepancy = 0
    
    # Return the yearly sales reports
    return {"success": True, "status_code": 200, "message": "Yearly sales reports retrieved successfully", "total_sales": total_sales, "total_discounts": total_discounts, "num_transactions": num_transactions, "gift_cards_issued": gift_cards_issued, "gift_cards_used": gift_cards_used, "discrepancy": discrepancy}

# Endpoint to get datewise sales reports
@auth_router.post("/get-datewise-sales-reports")
async def get_datewise_sales_reports(employee_code: str, room_id: str, date: str):
    # Check if the admin exists in the database
    admin = await users_collection.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")
    
    # Fetch sales data from the database
    sales_data = await sales_collection.find({"date": date}).to_list(None)
    
    # Calculate the total sales
    total_sales = 0
    for sale in sales_data:
        total_sales += sale["amount"]
    
    # Calculate the total discounts applied
    total_discounts = 0
    for sale in sales_data:
        total_discounts += sale["discount_amount"]
    
    # Calculate the number of transactions
    num_transactions = len(sales_data)
    
    # Calculate the gift cards issued and used
    gift_cards_issued = 0
    gift_cards_used = 0
    for sale in sales_data:
        if sale["gift_card_id"]:
            gift_cards_issued += 1
            gift_cards_used += 1
    
    # Calculate the discrepancy
    discrepancy = 0
    
    # Return the datewise sales reports
    return {"success": True, "status_code": 200, "message": "Datewise sales reports retrieved successfully", "total_sales": total_sales, "total_discounts": total_discounts, "num_transactions": num_transactions, "gift_cards_issued": gift_cards_issued, "gift_cards_used": gift_cards_used, "discrepancy": discrepancy}

# Endpoint to get low product alert
@auth_router.get("/get-low-product-alert")
async def get_low_product_alert(employee_code: str, room_id: str):
    # Fetch products from the database
    products = await products_collection.find().to_list(None)
    
    # Check if any product quantity is low
    low_products = []
    for product in products:
        if product["quantity"] < 5:
            low_products.append(product)
    
    # Return the low product alert
    if low_products:
        return {"success": True, "status_code": 200, "message": "Low product alert", "low_products": low_products}
    else:
        return {"success": False, "status_code": 404, "message": "No low products found"}

# Endpoint to restock product
@auth_router.put("/restock-product")
async def restock_product(product_id: str, quantity: int, employee_code: str, room_id: str):
    # Fetch product details from the database
    product = await products_collection.find_one({"id": product_id})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Update the product quantity
    await products_collection.update_one({"id": product_id}, {"$inc": {"quantity": quantity}})
    
    return {"success": True, "status_code": 200, "message": "Product restocked successfully"}
@auth_router.post("/handle-cash-shortage")
async def handle_cash_shortage(sales: SalesSchema):
    try:
        # Validate salesperson and room assignment
        salesperson = await users_collection.find_one({
            "employee_code": sales.employee_code,
            "role": "Salesperson"
        })
        if not salesperson:
            raise HTTPException(
                status_code=401,
                detail="Invalid employee code or unauthorized access"
            )

        room_assignment = await room_assignments_collection.find_one({
            "employee_code": sales.employee_code,
            "room_id": sales.room_id
        })
        if not room_assignment:
            raise HTTPException(
                status_code=403,
                detail="Salesperson is not assigned to this room"
            )

        # Get current cash register for the room
        cash_register = await cash_registers_collection.find_one({
            "room_id": sales.room_id,
            "date": sales.date
        })
        
        if not cash_register:
            # Initialize new cash register if not found
            cash_register = {
                "room_id": sales.room_id,
                "date": sales.date,
                "opening_balance": 0.0,
                "closing_balance": 0.0,
                "total_cash_sales": 0.0,
                "total_change_given": 0.0,
                "total_gift_cards_issued": 0.0,
                "total_toffees_issued": 0.0,
                "discrepancy": 0.0
            }
            await cash_registers_collection.insert_one(cash_register)

        # Calculate total amount needed
        total_amount = sales.cart.final_amount

        # Check if there's enough cash in register
        if cash_register["closing_balance"] < total_amount:
            shortage_amount = total_amount - sales.cash_received

            # Determine if gift card or toffee should be issued
            if shortage_amount >= 10:  # Threshold for gift card vs toffee
                # Issue gift card
                gift_card_number = f"GC-{sales.room_id}-{uuid.uuid4().hex[:8].upper()}"
                
                gift_card = {
                    "card_number": gift_card_number,
                    "value": shortage_amount,
                    "issued_date": sales.date,
                    "issued_by": sales.employee_code,
                    "room_id": sales.room_id,
                    "status": "active"
                }
                await gift_cards_collection.insert_one(gift_card)

                # Update cash register
                await cash_registers_collection.update_one(
                    {"_id": cash_register["_id"]},
                    {
                        "$inc": {
                            "total_gift_cards_issued": shortage_amount,
                            "closing_balance": -sales.cash_received
                        }
                    }
                )

                return {
                    "success": True,
                    "status_code": 200,
                    "message": "Gift card issued for shortage amount",
                    "data": {
                        "gift_card_number": gift_card_number,
                        "gift_card_value": shortage_amount,
                        "cash_received": sales.cash_received,
                        "date": sales.date
                    }
                }
            else:
                # Issue toffee
                toffee_value = shortage_amount
                
                # Update cash register
                await cash_registers_collection.update_one(
                    {"_id": cash_register["_id"]},
                    {
                        "$inc": {
                            "total_toffees_issued": toffee_value,
                            "closing_balance": -sales.cash_received
                        }
                    }
                )

                return {
                    "success": True,
                    "status_code": 200,
                    "message": "Toffee issued for shortage amount",
                    "data": {
                        "toffee_value": toffee_value,
                        "cash_received": sales.cash_received,
                        "date": sales.date
                    }
                }
        else:
            # Process normal change
            change_amount = sales.cash_received - total_amount

            # Update cash register
            await cash_registers_collection.update_one(
                {"_id": cash_register["_id"]},
                {
                    "$inc": {
                        "closing_balance": sales.cash_received,
                        "total_cash_sales": total_amount,
                        "total_change_given": change_amount
                    }
                }
            )

            return {
                "success": True,
                "status_code": 200,
                "message": "Transaction completed successfully",
                "data": {
                    "cash_received": sales.cash_received,
                    "change_given": change_amount,
                    "date": sales.date
                }
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}"
        )

# Helper endpoint to get cash register status
@auth_router.get("/get-cash-register-status/{room_id}")
async def get_cash_register_status(
    room_id: str,
    employee_code: str,
    date: str = datetime.datetime.now().strftime("%Y-%m-%d")
):
    try:
        # Validate salesperson and room assignment
        if not await validate_employee_room(employee_code, room_id):
            raise HTTPException(
                status_code=403,
                detail="Unauthorized access to room cash register"
            )

        cash_register = await cash_registers_collection.find_one({
            "room_id": room_id,
            "date": date
        })

        if not cash_register:
            return {
                "success": False,
                "status_code": 404,
                "message": "Cash register not found for this room and date"
            }

        return {
            "success": True,
            "status_code": 200,
            "data": {
                "opening_balance": cash_register["opening_balance"],
                "closing_balance": cash_register["closing_balance"],
                "total_cash_sales": cash_register["total_cash_sales"],
                "total_change_given": cash_register["total_change_given"],
                "total_gift_cards_issued": cash_register["total_gift_cards_issued"],
                "total_toffees_issued": cash_register["total_toffees_issued"],
                "discrepancy": cash_register["discrepancy"],
                "date": cash_register["date"]
            }
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching cash register status: {str(e)}"
        )

# Helper function to validate employee and room assignment
async def validate_employee_room(employee_code: str, room_id: str) -> bool:
    room_assignment = await room_assignments_collection.find_one({
        "employee_code": employee_code,
        "room_id": room_id
    })
    return bool(room_assignment)

@auth_router.post("/generate-barcode")
async def generate_barcode_endpoint(employee_code: str, num_barcodes: int):
    # Check if the admin exists in the database
    admin = await db.users.find_one({"employee_code": employee_code, "role": "Admin"})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid employee code or role")

    # Generate a list of barcodes
    barcodes = [generate_barcode() for _ in range(num_barcodes)]

    # Create a zip file to store the barcode images
    zip_file = zipfile.ZipFile("barcodes.zip", "w")

    # Create a barcode image for each barcode
    for i, barcode in enumerate(barcodes):
        # Create a barcode image
        barcode_image = Code128(barcode, writer=ImageWriter())
        barcode_image.save(f"barcode_{i+1}")

        # Add the barcode image to the zip file
        zip_file.write(f"barcode_{i+1}.png")

    # Close the zip file
    zip_file.close()

    # Return the zip file as a downloadable file
    return FileResponse("barcodes.zip", media_type="application/zip", filename="barcodes.zip")
