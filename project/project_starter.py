import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import pprint
import json
from openai import OpenAI
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from openai import OpenAI
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
# Load environment variables
dotenv.load_dotenv("config.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
 
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

agent_model = OpenAIServerModel(
    model_id=OPENAI_MODEL,
    api_base=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Base URL:", os.getenv("OPENAI_BASE_URL"))
print("Model:", os.getenv("OPENAI_MODEL"))

# Wrapper 1
@tool
def all_inventory_tool(as_of_date: str) -> dict:
    """
    Return all available inventory as of a given date.

    Args:
        as_of_date: date to retun the available inventory
    """
    try:
        inventory = get_all_inventory(as_of_date)
        return {
            "success": True,
            "as_of_date": as_of_date,
            "inventory": inventory
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def check_inventory(item_name: str, as_of_date: str) -> dict:
    """
    Check the current stock level for a specific paper item.
 
    Args:
        item_name: Name of the paper product.
        as_of_date: Date to check inventory in YYYY-MM-DD format.
 
    Returns:
        Inventory information including current stock.
    """
 
    # Map common customer/request wording to the inventory item names
    item_mapping = {
        "a4 glossy paper": "Glossy paper",
        "glossy a4 paper": "Glossy paper",
        "glossy paper": "Glossy paper",
 
        "a4 colored paper": "Colored paper",
        "colored a4 paper": "Colored paper",
        "colored paper": "Colored paper",
 
        "heavy cardstock": "Cardstock",
        "heavy cardstock (white)": "Cardstock",
        "cardstock": "Cardstock",
 
        "a4 white paper": "A4 paper",
        "white a4 paper": "A4 paper",
        "a4 printer paper": "A4 paper",
        "a4 paper": "A4 paper",
 
        "poster board": "Large poster paper (24x36 inches)",
        "poster boards": "Large poster paper (24x36 inches)",
        "large poster paper": "Large poster paper (24x36 inches)",
    }
 
    normalized_name = item_name.strip().lower()
 
    # Use mapped inventory name when available
    inventory_item_name = item_mapping.get(
        normalized_name,
        item_name.strip()
    )
 
    stock = get_stock_level(
        item_name=inventory_item_name,
        as_of_date=as_of_date
    )
 
    print("CHECK INVENTORY STOCK")
    print(stock)
 
    if stock.empty:
        return {
            "success": False,
            "item_name": inventory_item_name,
            "current_stock": 0,
            "message": "Item not found in inventory."
        }
 
    current_stock = int(stock.iloc[0]["current_stock"])
 
    return {
        "success": True,
        "item_name": inventory_item_name,
        "current_stock": current_stock
    }
 
# Wrapper 2
@tool
def check_supplier_delivery(input_date_str: str, quantity: int, requested_delivery_date: str) -> dict:
    """
    Determine the expected supplier delivery date.
 
    Args:
        input_date_str: Starting date in YYYY-MM-DD format.
        quantity: Quantity that must be supplied.
        requested_delivery_date: Customer requested delivery date in YYYY-MM-DD format.
 
    Returns:
        Expected supplier delivery information.
    """
    delivery_date = get_supplier_delivery_date(
        input_date_str=input_date_str,
        quantity=quantity
    )
 
    supplier_date = pd.to_datetime(delivery_date)
    requested_date = pd.to_datetime(requested_delivery_date)
 
    can_deliver_in_time = supplier_date <= requested_date
 
    return {
        "success": True,
        "delivery_date": str(delivery_date),
        "requested_delivery_date": requested_delivery_date,
        "can_deliver_in_time": bool(can_deliver_in_time)
    }


# Database and model configuration will be shared
# across all specialized agents.
"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

@tool
def quote_history_tool(
    search_terms: List[str],
    limit: int = 5
) -> dict:
    """
    Find relevant historical customer quotes.
 
    Args:
        search_terms: Terms to use when searching historical customer quotes.
        limit: Maximum number of quote-history results to return.
 
    Returns:
        A dictionary containing historical quote records.
    """
    try:
        quotes = search_quote_history(
            search_terms=search_terms,
            limit=limit
        )
 
        # Convert DataFrame results into records that the agent
        # can read reliably.
        if isinstance(quotes, pd.DataFrame):
            quote_records = quotes.to_dict(orient="records")
        elif isinstance(quotes, list):
            quote_records = quotes
        else:
            quote_records = []
 
        # Extract available historical prices explicitly
        historical_prices = []
 
        for quote in quote_records:
            if isinstance(quote, dict):
                amount = quote.get("total_amount")
 
                if amount is not None:
                    try:
                        amount = float(amount)
 
                        if amount > 0:
                            historical_prices.append(amount)
 
                    except (TypeError, ValueError):
                        pass
 
        return {
            "success": True,
            "search_terms": search_terms,
            "quote_count": len(quote_records),
            "quotes": quote_records,
            "historical_prices": historical_prices,
            "has_pricing_data": len(historical_prices) > 0
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "quote_count": 0,
            "quotes": [],
            "historical_prices": [],
            "has_pricing_data": False
        }
 
@tool
def cash_balance_tool(as_of_date: str) -> dict:
    """
    Get the company's cash balance for a specified date.
 
    Args:
        as_of_date: Date for which to retrieve the cash balance, in YYYY-MM-DD format.
 
    Returns:
        A dictionary containing the cash balance information.
    """
 
    try:
        balance = get_cash_balance(as_of_date)
 
        return {
            "success": True,
            "as_of_date": as_of_date,
            "cash_balance": balance
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def financial_report_tool(as_of_date: str) -> dict:
    """
    Get a financial report for the requested date.
 
    Args:
        as_of_date: Date for which to generate the financial report, in YYYY-MM-DD format.
 
    Returns:
        A dictionary containing the requested financial report.
    """
 
    try:
        report = generate_financial_report(as_of_date)
 
        return {
            "success": True,
            "report": report
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def fulfill_order_tool(
    item_name: str,
    quantity: int,
    total_price: float,
    sale_date: str
) -> dict:
    """
    Record a completed customer sale.
 
    Args:
        item_name: Name of the item being sold.
        quantity: Quantity of the item being sold.
        total_price: Total price of the completed sale.
        sale_date: Date of the sale in YYYY-MM-DD format.
 
    Returns:
        A dictionary containing the order fulfillment result.
    """
 
    try:
        # A fulfilled order must always have a positive quoted price
        if total_price <= 0:
            return {
                "success": False,
                "message": (
                    "Cannot fulfill order because total_price must "
                    "be greater than 0."
                )
            }
 
        # Normalize common customer descriptions to inventory item names
        item_mapping = {
            "a4 glossy paper": "Glossy paper",
            "glossy a4 paper": "Glossy paper",
            "glossy paper": "Glossy paper",
 
            "colored paper": "Colored paper",
            "colorful paper": "Colored paper",
            "coloured paper": "Colored paper",
 
            "heavy cardstock": "Cardstock",
            "heavy cardstock (white)": "Cardstock",
            "cardstock": "Cardstock",
 
            "a4 paper": "A4 paper",
            "white a4 paper": "A4 paper",
            "matte a4 paper": "A4 paper",
 
            "poster board": "Large poster paper (24x36 inches)",
            "poster boards": "Large poster paper (24x36 inches)",
            "poster board paper": "Large poster paper (24x36 inches)"
        }
 
        normalized_name = item_name.strip().lower()
 
        inventory_item_name = item_mapping.get(
            normalized_name,
            item_name.strip()
        )
 
        # Confirm that the item exists in inventory.
        #
        # Do NOT reject here only because current stock is below
        # the requested quantity. The Inventory Agent has already
        # determined whether supplier replenishment can arrive
        # before the requested delivery date.
        stock_df = get_stock_level(
            inventory_item_name,
            sale_date
        )
 
        if stock_df.empty:
            return {
                "success": False,
                "message": "Item not found.",
                "item_name": inventory_item_name
            }
 
        # Record the completed sale
        transaction_id = create_transaction(
            item_name=inventory_item_name,
            transaction_type="sales",
            quantity=quantity,
            price=total_price,
            date=sale_date
        )
 
        return {
            "success": True,
            "transaction_id": transaction_id,
            "item_name": inventory_item_name,
            "quantity": quantity,
            "total_price": total_price
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Tools for ordering agent

## Workers
inventory_worker = ToolCallingAgent(
    tools=[
        check_inventory,
        check_supplier_delivery,
        all_inventory_tool
    ],
    model=agent_model,
    name="inventory_agent",
    description=(
        "Inventory management worker. Checks current paper inventory "
        "and determines whether supplier replenishment is required."
    ),
    instructions="""
You are the Inventory Agent.
 
Always check current inventory before making an inventory decision.
 
For each requested item:
 
1. Use check_inventory to determine the current stock.
 
2. If current stock is greater than or equal to the requested quantity:
   - shortage = 0
   - requires_reorder = false
   - delivery_date = null
   - status = available
   - can_fulfill = true
 
3.If current stock is less than the requested quantity:
- Calculate shortage = requested quantity - current_stock.
- Call check_supplier_delivery using exactly the shortage quantity.
- Compare the returned supplier delivery_date with the customer's requested delivery date.
 
CRITICAL RULE:
Current stock being 0 or insufficient does NOT automatically mean the order
cannot be fulfilled.
 
IMPORTANT DATE COMPARISON RULE:
 
Always compare supplier_delivery_date and requested_delivery_date
as actual calendar dates in YYYY-MM-DD format.
 
If supplier_delivery_date <= requested_delivery_date:
- The shortage CAN be replenished in time.
- Treat inventory as sufficient for fulfillment by the requested date.
- status = available_by_requested_date
- requires_reorder = true
- Tell the manager explicitly that the order CAN be fulfilled by
  the requested delivery date.
- Do NOT reject the order simply because current_stock is 0.
 
If supplier_delivery_date > requested_delivery_date:
- The shortage cannot be replenished in time.
- status = unavailable_by_requested_date
- requires_reorder = true
- Tell the manager explicitly that the order CANNOT be fulfilled
  by the requested delivery date.
 
Always compare the dates chronologically.
For example:
2023-10-14 is before 2025-04-15.
Therefore, supplier delivery on 2023-10-14 CAN satisfy an order required
by 2025-04-15.
 
 
4. Compare the supplier delivery date with the customer's required delivery date:
   - If supplier delivery date is on or before the customer's required date,
     status = available_after_reorder
     can_fulfill = true
   - If supplier delivery date is after the customer's required date,
     status = cannot_fulfill_by_requested_date
     can_fulfill = false
 
5. Do not treat a reorder by itself as a failure.
   A reorder is acceptable when the supplier can deliver the shortage
   on or before the customer's required delivery date.
 
Return a clear inventory result to the manager agent including:
- item_name
- requested_quantity
- current_stock
- shortage
- requires_reorder
- delivery_date
- can_fulfill
- status
"""
)


quoting_worker = ToolCallingAgent(
    tools=[
        quote_history_tool,
        cash_balance_tool,
        financial_report_tool
    ],
    model=agent_model,
    name="quoting_agent",
    description=(
        "Quoting worker. Reviews historical quotes and financial "
        "information to determine an appropriate customer quote."
    ),
    instructions="""
You are the Quoting Agent.
 
Your responsibility is customer pricing.

For customer sales, do not reject a quote solely because the company's
cash balance is zero. Cash balance should only be considered when
additional inventory must be purchased.
 
If sufficient inventory is already available, use historical quote data
to determine a reasonable unit price and calculate a positive total price.
 
Use the available quoting and financial tools when needed.
Consider historical quotes and relevant financial information.

HISTORICAL PRICING RULES:
 
You MUST make multiple pricing attempts before concluding that pricing
is unavailable.
 
For each requested paper item:
 
1. First search using the customer's exact item description.
 
2. If no historical pricing is returned, immediately search using a
   simpler canonical description.
 
Examples:
- "A4 glossy paper" -> search "glossy paper"
- "heavy cardstock (white)" -> search "cardstock"
- "colored paper (assorted colors)" -> search "colored paper"
- "A4 paper" -> search "A4 paper"
- "poster board" -> search "poster paper"
 
3. If that search still returns no pricing, broaden the search again
   using "paper".
 
4. If quote_history_tool returns:
       has_pricing_data = True
   you MUST use one of the returned historical_prices to determine
   a reasonable positive unit_price.
 
5. Do NOT reject pricing merely because:
   - current_stock is 0,
   - cash_balance is 0 or low,
   - supplier replenishment is required.
 
Inventory feasibility is handled by the Inventory Agent.
Cash balance does not prevent quoting a customer sale.
 
6. Only state that historical pricing is unavailable AFTER all relevant
   exact and broader searches have been attempted.
 
7. Never return total_price = 0 for an order when usable historical
   pricing data has been found.
 
Calculate:
 
    total_price = unit_price * requested_quantity
 
The returned total_price must be greater than 0.

CRITICAL:
 
If the Inventory Agent has already determined that inventory can be
available by the customer's requested delivery date, DO NOT reconsider
or reject the order because current_stock is 0.
 
Your responsibility is pricing only.
 
If historical pricing is found, return a positive total_price to the
Orchestrator so that it can continue to the Sales Agent.

IMPORTANT PRICING RULES:
 
For customer sales, DO NOT reject or refuse to provide a quote
because the company's cash balance is zero.
 
Cash balance is only relevant when additional inventory must be purchased.
 
You MUST attempt to determine a positive unit price using historical quotes.
 
If an exact historical match is unavailable:
1. Search using broader item terms.
2. Use the closest relevant historical quote.
3. Use its historical price as the pricing reference.
 
Never return a price of 0 for an order that can otherwise be fulfilled.
 
For every order that can be fulfilled:
total_price = unit_price * quantity
 
The returned total_price MUST be greater than 0.
"""
)

sales_worker = ToolCallingAgent(
    tools=[
        fulfill_order_tool
    ],
    model=agent_model,
    name="sales_agent",
    description=(
        "Sales finalization worker. Completes approved customer orders "
        "and records successful sales."
    ),
    instructions="""
You are the Sales Agent.
 
Finalize a sale only after the manager has determined that the
order can be fulfilled and pricing has been established.

IMPORTANT:
The manager's fulfillment decision is authoritative.
 
If the manager tells you that the order can be fulfilled by the
customer's requested delivery date, proceed with the sale even if
the current inventory is insufficient or zero.
 
Do NOT reject an order solely because current_stock is less than
the requested quantity when supplier replenishment has already
been confirmed in time.
 
Supplier replenishment confirmed by the inventory agent counts as
sufficient inventory for fulfillment.
 
Only reject the sale for insufficient inventory when the manager
explicitly states that the order cannot be fulfilled by the
requested delivery date.
 
The total_price MUST be greater than 0.
Use the total quoted price supplied by the manager.
Never invent a price and never use 0.0 for a fulfilled sale.
 
Call fulfill_order_tool with:
- item_name
- quantity
- total_price: the complete quoted price for the order
- sale_date
 
Use fulfill_order_tool to record the completed sale.
Return the transaction result to the manager agent.
"""
)

# Orchestrator Agent

orchestrator_agent = ToolCallingAgent(
    tools=[],
    model=agent_model,
    managed_agents=[
        inventory_worker,
        quoting_worker,
        sales_worker
    ],
    name="orchestrator_agent",
    description="Coordinates the paper company's customer order workflow.",
    instructions="""
You are the Orchestrator Agent for the paper company.
 
You manage customer order requests by delegating work to your
specialized managed agents.
 
Use the Inventory Agent for:
- checking inventory
- determining shortages
- determining supplier delivery needs
 
Use the Quoting Agent for:
- historical quote research
- pricing
- financial considerations
 
Use the Sales Agent for:
- finalizing fulfillable orders
- recording completed sales
 
IMPORTANT DATE RULE:
 
IMPORTANT DATE RULE:
 
When inventory requires supplier replenishment, compare the supplier
delivery date with the customer's requested delivery date.
 
If supplier_delivery_date <= requested_delivery_date:
- The replenishment arrives in time.
- Treat the required inventory as available by the requested date.
- Continue to the Quoting Agent.
- If a valid total_price greater than 0 is obtained, continue to the Sales Agent.
 
If supplier_delivery_date > requested_delivery_date:
- The replenishment arrives too late.
- Do not fulfill the order.
 
Always compare dates chronologically.
For example, 2023-10-14 is before 2025-04-15, so inventory arriving
on 2023-10-14 IS available in time for an order required by 2025-04-15.
 
CRITICAL FULFILLMENT RULES:
 
Do not reject an order merely because current_stock is insufficient.
 
If current_stock is less than the requested quantity:
- Determine the shortage.
- Ask the Inventory Agent for the supplier_delivery_date.
- If supplier_delivery_date <= requested_delivery_date, treat the shortage
  as available in time and continue to the Quoting Agent.
- If supplier_delivery_date > requested_delivery_date, the order cannot
  be fulfilled.
 
IMPORTANT CASH RULE:
 
A cash balance of 0 or a low cash balance does NOT automatically make a
customer order unfulfillable.
 
Do NOT reject a customer sale merely because cash_balance is 0 or because
the company currently has insufficient cash.
 
If existing inventory OR supplier replenishment arriving on or before the
requested delivery date can satisfy the order, continue to the Quoting Agent.
 
The Quoting Agent must attempt to obtain a positive total_price.
 
If a valid total_price > 0 is obtained, continue to the Sales Agent and
fulfill the order.
 
Only reject the order when:
- the required inventory cannot be available by the requested delivery date, or
- no valid positive price can be determined after historical pricing attempts.
 
For a customer order:
1. Determine the requested paper item, quantity, and request date.
2. Delegate inventory analysis to the Inventory Agent.
3. If sufficient inventory is available, delegate pricing to the Quoting Agent.
4. Obtain a numeric total_price greater than 0 from the Quoting Agent.
5. Only after inventory and pricing are confirmed, delegate the sale to the Sales Agent.
6. Give the Sales Agent the item name, quantity, request date, and quoted total_price.
7. Never fulfill an order with total_price equal to 0.
8. Return a clear final response describing whether the order was fulfilled and why.
 
Do not perform worker responsibilities yourself when an appropriate
managed agent is available.
"""
)
 
# Run your test scenarios by writing them here. Make sure to keep track of them.
def run_test_scenarios():
    """
    Run all requests from quote_requests_sample.csv
    through the orchestrator and validate evaluation requirements.
    """
 
    df = pd.read_csv("quote_requests_sample.csv")
 
    print(f"\nLoaded {len(df)} requests from quote_requests_sample.csv")
 
    results = []
 
    for index, request_row in df.iterrows():
 
        print("\n" + "=" * 60)
        print(f"REQUEST {index + 1}")
        print("=" * 60)
        print(request_row["request"])
 
        # -----------------------------------------------------
        # Capture number of SALES before this request
        # -----------------------------------------------------
        with db_engine.connect() as conn:
            sales_before = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM transactions
                    WHERE transaction_type = 'sales'
                """)
            ).scalar()
 
        # -----------------------------------------------------
        # Capture cash before this request
        # -----------------------------------------------------
        cash_before = get_cash_balance(datetime.now())
 
        # -----------------------------------------------------
        # Run request through orchestrator
        # -----------------------------------------------------
        try:
            result = orchestrator_agent.run(
                f"""
Customer request:
 
{str(request_row["request"])}
 
Request date: {str(request_row["request_date"])}
 
Process this request through the appropriate agents.
 
Check inventory first.
 
If sufficient inventory is available, obtain a valid positive quote
and fulfill the order.
 
If inventory is insufficient, determine whether supplier replenishment
can arrive on or before the requested delivery date.
 
If replenishment can arrive in time, treat the required inventory as
available and continue to pricing.
 
If a valid positive price is obtained, send the order to the Sales Agent
and record the completed sale.
 
Do not reject the order merely because current inventory is insufficient
or because the current cash balance is low or zero.
 
If the request cannot be fulfilled, clearly explain the reason.
"""
            )
 
            result_text = str(result)
 
        except Exception as e:
            # Do not stop processing the remaining requests
            result_text = f"Request could not be fulfilled because: {str(e)}"
 
            print("\nERROR PROCESSING REQUEST:")
            print(result_text)
 
        # -----------------------------------------------------
        # Capture number of SALES after this request
        # -----------------------------------------------------
        with db_engine.connect() as conn:
            sales_after = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM transactions
                    WHERE transaction_type = 'sales'
                """)
            ).scalar()
 
        # -----------------------------------------------------
        # Capture cash after this request
        # -----------------------------------------------------
        cash_after = get_cash_balance(datetime.now())
 
        # -----------------------------------------------------
        # Determine fulfillment
        # -----------------------------------------------------
        sales_created = int(sales_after) - int(sales_before)
 
        fulfilled = sales_created > 0
 
        cash_changed = cash_before != cash_after
 
        # Since a completed customer sale changes cash,
        # use the verified cash difference for this evaluation.
        total_price = (
            abs(float(cash_after) - float(cash_before))
            if fulfilled
            else 0.0
        )
 
        # Clear reason for rejected/unfulfilled request
        if fulfilled:
            unfulfilled_reason = ""
        else:
            unfulfilled_reason = result_text.strip()
 
        # -----------------------------------------------------
        # Store result
        # -----------------------------------------------------
        results.append({
            "request_id": index + 1,
            "job": request_row["job"],
            "need_size": request_row["need_size"],
            "event": request_row["event"],
            "request": request_row["request"],
            "request_date": request_row["request_date"],
 
            "cash_before": cash_before,
            "cash_balance": cash_after,
            "cash_changed": cash_changed,
 
            "sales_created": sales_created,
 
            "fulfilled": fulfilled,
            "total_price": total_price,
 
            "result": result_text,
            "unfulfilled_reason": unfulfilled_reason
        })
 
        print("\n--- REQUEST RESULT ---")
        print(f"Sales before: {sales_before}")
        print(f"Sales after: {sales_after}")
        print(f"Sales created: {sales_created}")
        print(f"Fulfilled: {fulfilled}")
        print(f"Cash before: {cash_before}")
        print(f"Cash after: {cash_after}")
        print(f"Cash changed: {cash_changed}")
        print(f"Total price: {total_price}")
 
    # =========================================================
    # Create final dataframe
    # =========================================================
    results_df = pd.DataFrame(results)
 
    # Save BEFORE assertions so results are available
    # even if one validation requirement fails.
    results_df.to_csv(
        "test_results.csv",
        index=False
    )
 
    print("\n" + "=" * 60)
    print("FULFILLMENT SUMMARY")
    print("=" * 60)
 
    print(
        results_df[
            [
                "request_id",
                "fulfilled",
                "sales_created",
                "total_price",
                "cash_before",
                "cash_balance",
                "cash_changed"
            ]
        ].to_string(index=False)
    )
 
    # =========================================================
    # Reviewer validation
    # =========================================================
 
    total_requests = len(results_df)
 
    fulfilled_count = int(
        results_df["fulfilled"].sum()
    )
 
    cash_change_count = int(
        results_df["cash_changed"].sum()
    )
 
    unfulfilled_with_reason = results_df[
        (results_df["fulfilled"] == False)
        &
        (
            results_df["unfulfilled_reason"]
            .fillna("")
            .str.strip()
            .ne("")
        )
    ]
 
    print("\n" + "=" * 60)
    print("REVIEWER VALIDATION SUMMARY")
    print("=" * 60)
 
    print(f"Total requests processed: {total_requests}")
    print(f"Fulfilled requests: {fulfilled_count}")
    print(f"Rows where cash balance changed: {cash_change_count}")
    print(
        "Unfulfilled requests with reason: "
        f"{len(unfulfilled_with_reason)}"
    )
 
    # =========================================================
    # Required reviewer conditions
    # =========================================================
 
    assert total_requests == 20, (
        f"Expected 20 processed requests, "
        f"but got {total_requests}"
    )
 
    assert fulfilled_count >= 3, (
        f"Expected at least 3 fulfilled requests, "
        f"but got {fulfilled_count}"
    )
 
    assert cash_change_count >= 3, (
        f"Expected at least 3 rows where cash balance changes, "
        f"but got {cash_change_count}"
    )
 
    assert len(unfulfilled_with_reason) >= 1, (
        "Expected at least one unfulfilled request "
        "with a clear reason."
    )
 
    print("\nALL REVIEWER VALIDATION CHECKS PASSED")
 
    print("\nUnfulfilled requests:")
    print(
        unfulfilled_with_reason[
            [
                "request_id",
                "request",
                "unfulfilled_reason"
            ]
        ].to_string(index=False)
    )
 
    print("\nEvaluation completed.")
    print("Results saved to test_results.csv")

if __name__ == "__main__":
    init_database(db_engine)
 
    run_test_scenarios()
