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
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Base URL:", os.getenv("OPENAI_BASE_URL"))
print("Model:", os.getenv("OPENAI_MODEL"))

def run_llm_agent(system_prompt, user_request, tools, tool_functions):
    """
    Run an LLM-powered agent.
 
    The LLM decides which available tool to call based on
    the request and the results returned by previous tool calls.
    """
 
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_request
        }
    ]
 
    while True:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
 
        message = response.choices[0].message
        messages.append(message)
 
        # No tool call means the LLM has finished its task
    if not message.tool_calls:
        content = message.content
 
    try:
        result = json.loads(content)
 
        # Handle a response that was JSON encoded twice
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                pass
 
        return result
 
    except (json.JSONDecodeError, TypeError):
        return {
            "status": "completed",
            "message": content
        }
 
        # Execute only the tools selected by the LLM
        for tool_call in message.tool_calls:
 
            tool_name = tool_call.function.name
 
            arguments = json.loads(
                tool_call.function.arguments
            )
 
            print(f"\nLLM selected tool: {tool_name}")
            print(f"Tool arguments: {arguments}")

            if tool_name not in tool_functions:
                result = {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                result = tool_functions[tool_name](**arguments)
 
            # Give the tool result back to the LLM
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        default=str
                    )
                }
            )
# ---------------------------------------------------------
# Inventory Agent - LLM Tool Definitions
# ---------------------------------------------------------
 
INVENTORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": (
                "Check the current stock level for a specific item "
                "as of a given date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the paper product"
                    },
                    "as_of_date": {
                        "type": "string",
                        "description": "Date to check inventory in YYYY-MM-DD format"
                    }
                },
                "required": ["item_name", "as_of_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_supplier_delivery",
            "description": (
                "Determine the expected supplier delivery date "
                "when additional inventory is required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_date_str": {
                        "type": "string",
                        "description": "Request date in YYYY-MM-DD format"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity of additional stock required"
                    }
                },
                "required": ["input_date_str", "quantity"]
            }
        }
    }
]

# Wrapper 1
def check_inventory(item_name, as_of_date):
    stock = get_stock_level(
        item_name=item_name,
        as_of_date=as_of_date
    )
 
    return {
        "success": True,
        "item_name": item_name,
        "current_stock": stock
    }
 
# Wrapper 2
def check_supplier_delivery(input_date_str, quantity):
    delivery_date = get_supplier_delivery_date(
        input_date_str=input_date_str,
        quantity=quantity
    )
 
    return {
        "success": True,
        "delivery_date": str(delivery_date)
    }

# MApping
INVENTORY_TOOL_FUNCTIONS = {
    "check_inventory": check_inventory,
    "check_supplier_delivery": check_supplier_delivery
}

# Database and model configuration will be shared
# across all specialized agents.
"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
def inventory_lookup_tool(item_name: str, as_of_date: str) -> dict:
    """
    Check the current stock level for a specific paper item.
    """
    try:
        stock_df = get_stock_level(item_name, as_of_date)
 
        if stock_df.empty:
            return {
                "success": False,
                "item_name": item_name,
                "message": "Item not found in inventory."
            }
 
        current_stock = int(stock_df.iloc[0]["current_stock"])
 
        return {
            "success": True,
            "item_name": item_name,
            "current_stock": current_stock,
            "as_of_date": as_of_date
        }
 
    except Exception as e:
        return {
            "success": False,
            "item_name": item_name,
            "error": str(e)
        }

def all_inventory_tool(as_of_date: str) -> dict:
    """
    Return stock levels for all paper products.
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
def supplier_delivery_tool(
    request_date: str,
    quantity: int
) -> dict:
    """
    Estimate when replenishment stock can arrive.
    """
    try:
        delivery_date = get_supplier_delivery_date(
            request_date,
            quantity
        )
 
        return {
            "success": True,
            "quantity": quantity,
            "request_date": request_date,
            "delivery_date": delivery_date
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def reorder_inventory_tool(
    item_name: str,
    quantity: int,
    unit_price: float,
    order_date: str
) -> dict:
    """
    Create a stock order transaction to replenish inventory.
    """
 
    try:
        total_price = quantity * unit_price
 
        transaction_id = create_transaction(
            item_name=item_name,
            transaction_type="stock_orders",
            quantity=quantity,
            price=total_price,
            date_str=order_date
        )
 
        delivery_date = get_supplier_delivery_date(
            order_date,
            quantity
        )
 
        return {
            "success": True,
            "transaction_id": transaction_id,
            "item_name": item_name,
            "quantity": quantity,
            "total_price": total_price,
            "delivery_date": delivery_date
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Tools for quoting agent
def quote_history_tool(
    search_terms: List[str],
    limit: int = 5
) -> dict:
    """
    Find relevant historical customer quotes.
    """
 
    try:
        quotes = search_quote_history(
            search_terms=search_terms,
            limit=limit
        )
 
        return {
            "success": True,
            "search_terms": search_terms,
            "quotes": quotes
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def cash_balance_tool(as_of_date: str) -> dict:
    """
    Get the company's available cash balance.
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

def financial_report_tool(as_of_date: str) -> dict:
    """
    Generate the company's financial and inventory report.
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

def fulfill_order_tool(
    item_name: str,
    quantity: int,
    total_price: float,
    sale_date: str
) -> dict:
    """
    Record a completed customer sale.
    """
 
    try:
        stock_df = get_stock_level(
            item_name,
            sale_date
        )
 
        if stock_df.empty:
            return {
                "success": False,
                "message": "Item not found."
            }
 
        available_stock = int(
            stock_df.iloc[0]["current_stock"]
        )
 
        if available_stock < quantity:
            return {
                "success": False,
                "message": "Insufficient inventory.",
                "available_stock": available_stock,
                "requested_quantity": quantity
            }
 
        transaction_id = create_transaction(
            item_name=item_name,
            transaction_type="sales",
            quantity=quantity,
            price=total_price,
            date=sale_date
        )
 
        return {
            "success": True,
            "transaction_id": transaction_id,
            "item_name": item_name,
            "quantity": quantity,
            "total_price": total_price
        }
 
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Tools for ordering agent


# Set up your agents and create an orchestration agent that will manage them.
# Inventory Agent-
def inventory_agent(
    item_name: str,
    quantity: int,
    request_date: str
):
    """
    LLM-powered Inventory Agent.
 
    The LLM decides which inventory tools to call based on
    the request and the results returned by the tools.
    """
    system_prompt = """
2. Compare the current_stock returned by check_inventory with
   the requested quantity.
 
3. If current_stock is greater than or equal to the requested
   quantity:
   - DO NOT call check_supplier_delivery.
   - Set shortage to 0.
   - Set requires_reorder to false.
   - Set delivery_date to null.
   - Set status to "available".
 
4. Only if current_stock is less than the requested quantity:
   - Calculate shortage as requested quantity minus current_stock.
   - Call check_supplier_delivery using exactly that shortage
     quantity.
   - Set requires_reorder to true.
   - Set status to "reorder_required".
 
5. NEVER call check_supplier_delivery with quantity 0.
"""

    user_request = f"""
    Item name: {item_name}
    Requested quantity: {quantity}
    Request date: {request_date}
 
    Determine whether this request can be fulfilled from inventory.
    """
    return run_llm_agent(
        system_prompt=system_prompt,
        user_request=user_request,
        tools=INVENTORY_TOOLS,
        tool_functions=INVENTORY_TOOL_FUNCTIONS
    )


 # ALM Agent

def run_llm_agent(system_prompt, user_request, tools, tool_functions):
    """
    Runs an LLM-powered agent and allows the LLM
    to dynamically decide which available tool to use.
    """
 
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_request
        }
    ]
 
    while True:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
 
        message = response.choices[0].message
        messages.append(message)
 
        # If the LLM does not request another tool,
        # the agent has finished.
        if not message.tool_calls:
            return message.content
 
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(
                tool_call.function.arguments
            )
 
            if tool_name not in tool_functions:
                result = {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                result = tool_functions[tool_name](**arguments)
 
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str)
                }
            )

# ---------------------------------------------------------
# Quote Agent - LLM Tool Definitions
# ---------------------------------------------------------
 
QUOTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_quote_history",
            "description": (
                "Search historical customer quotes for similar paper "
                "products and requests. Use the results to help prepare "
                "an appropriate customer quote."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Terms to use when searching historical quotes, "
                            "such as the item name and customer request."
                        )
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of historical quotes to return"
                    }
                },
                "required": ["search_terms"]
            }
        }
    }
]
# Wrapper
def search_quote_history(search_terms, limit=5):
    return quote_history_tool(
        search_terms=search_terms,
        limit=limit
    )

# MAppings
QUOTE_TOOL_FUNCTIONS = {
    "search_quote_history": search_quote_history
}

def quote_agent(
    customer_request: str,
    item_name: str,
    quantity: int,
    request_date: str,
    inventory_result: dict
):
    """
    LLM-powered Quote Agent.
 
    The LLM decides when historical quote information is needed
    and uses the available quote tools to prepare the quote.
    """
 
    system_prompt = """
You are the Quote Agent for a paper supply company.
 
Your responsibility is to prepare customer quote information.
 
You have access to a tool that searches historical quotes.
 
Follow these rules:
 
1. Review the customer's request, item, quantity, request date,
   and inventory information.
 
2. Use the search_quote_history tool when historical quote
   information would help prepare the quote.
 
3. Choose useful search terms based on the customer's request
   and item name.
 
4. Use the historical quote results when preparing the response.
 
5. Do not check or modify inventory yourself. Inventory
   information is supplied to you.
 
6. Do not fulfill or record a sale.
 
7. Do not invent historical quote information. Use the tool
   when historical information is needed.
 
Return the final result as valid JSON only.
 
Use this structure:
 
{
    "agent": "quote_agent",
    "status": "quote_prepared",
    "item_name": "product name",
    "quantity": 0,
    "request_date": "YYYY-MM-DD",
    "inventory_result": {},
    "historical_quotes": [],
    "message": "quote information"
}
"""
 
    user_request = f"""
Customer request: {customer_request}
 
Item name: {item_name}
Quantity: {quantity}
Request date: {request_date}
 
Inventory information:
{json.dumps(inventory_result, default=str)}
 
Prepare the quote information for this customer.
"""
 
    return run_llm_agent(
        system_prompt=system_prompt,
        user_request=user_request,
        tools=QUOTE_TOOLS,
        tool_functions=QUOTE_TOOL_FUNCTIONS
    )

# ---------------------------------------------------------
# Sales Agent - LLM Tool Definitions
# ---------------------------------------------------------
 
SALES_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fulfill_order",
            "description": (
                "Fulfill and record a customer order. "
                "Use this tool only when the supplied inventory "
                "information confirms that the order can be fulfilled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the paper product"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity being purchased"
                    },
                    "total_price": {
                        "type": "number",
                        "description": "Total price of the order"
                    },
                    "sale_date": {
                        "type": "string",
                        "description": "Date of the sale in YYYY-MM-DD format"
                    }
                },
                "required": [
                    "item_name",
                    "quantity",
                    "total_price",
                    "sale_date"
                ]
            }
        }
    }
]

# Wrapper
def fulfill_customer_order(
    item_name,
    quantity,
    total_price,
    sale_date
):
    return fulfill_order_tool(
        item_name=item_name,
        quantity=quantity,
        total_price=total_price,
        sale_date=sale_date
    )
# Mappings
SALES_TOOL_FUNCTIONS = {
    "fulfill_order": fulfill_customer_order
}

def sales_agent(
    item_name: str,
    quantity: int,
    total_price: float,
    request_date: str,
    inventory_result: dict
):
    """
    LLM-powered Sales Agent.
 
    The LLM reviews the inventory information and decides
    whether the order fulfillment tool should be called.
    """
 
    system_prompt = """
You are the Sales Agent for a paper supply company.
 
Your responsibility is to determine whether a customer order
can be fulfilled and, when appropriate, record the sale.
 
You have access to the fulfill_order tool.
 
Follow these rules:
 
1. Carefully review the supplied inventory information.
 
2. If the inventory information confirms that sufficient stock
   is currently available and no reorder is required, use the
   fulfill_order tool to complete the order.
 
3. If inventory is insufficient, requires a reorder, or cannot
   currently satisfy the requested quantity, DO NOT call the
   fulfill_order tool.
 
4. You must decide whether the fulfill_order tool should be
   called based on the supplied information.
 
5. Never invent inventory availability.
 
6. Never fulfill an order when the inventory information says
   additional inventory is required.
 
Return the final result as valid JSON only.
 
For a completed order, return information similar to:
 
{
    "agent": "sales_agent",
    "status": "completed",
    "message": "Order successfully fulfilled"
}
 
For an order that cannot currently be fulfilled, return:
 
{
    "agent": "sales_agent",
    "status": "pending_inventory",
    "message": "Order cannot currently be fulfilled",
    "expected_delivery_date": null
}
"""
 
    user_request = f"""
Item name: {item_name}
Quantity: {quantity}
Total price: {total_price}
Request date: {request_date}
 
Inventory information:
{json.dumps(inventory_result, default=str)}
 
Determine whether this order can be fulfilled.
If appropriate, use the available fulfillment tool.
"""
 
    return run_llm_agent(
        system_prompt=system_prompt,
        user_request=user_request,
        tools=SALES_TOOLS,
        tool_functions=SALES_TOOL_FUNCTIONS
    )
 

# ---------------------------------------------------------
# Orchestrator - Agent Delegation Wrappers
# ---------------------------------------------------------
 
def delegate_to_inventory_agent(
    item_name,
    quantity,
    request_date
):
    return inventory_agent(
        item_name=item_name,
        quantity=quantity,
        request_date=request_date
    )
 
 
def delegate_to_quote_agent(
    customer_request,
    item_name,
    quantity,
    request_date,
    inventory_result
):
    return quote_agent(
        customer_request=customer_request,
        item_name=item_name,
        quantity=quantity,
        request_date=request_date,
        inventory_result=inventory_result
    )
 
 
def delegate_to_sales_agent(
    item_name,
    quantity,
    total_price,
    request_date,
    inventory_result
):
    return sales_agent(
        item_name=item_name,
        quantity=quantity,
        total_price=total_price,
        request_date=request_date,
        inventory_result=inventory_result
    )
# Tool definition
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inventory_agent",
            "description": (
                "Delegate inventory checking and supplier delivery "
                "decisions to the Inventory Agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string"
                    },
                    "quantity": {
                        "type": "integer"
                    },
                    "request_date": {
                        "type": "string"
                    }
                },
                "required": [
                    "item_name",
                    "quantity",
                    "request_date"
                ]
            }
        }
    },
 
    {
        "type": "function",
        "function": {
            "name": "quote_agent",
            "description": (
                "Delegate quote preparation and historical quote "
                "analysis to the Quote Agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_request": {
                        "type": "string"
                    },
                    "item_name": {
                        "type": "string"
                    },
                    "quantity": {
                        "type": "integer"
                    },
                    "request_date": {
                        "type": "string"
                    },
                    "inventory_result": {
                        "type": "object"
                    }
                },
                "required": [
                    "customer_request",
                    "item_name",
                    "quantity",
                    "request_date",
                    "inventory_result"
                ]
            }
        }
    },
 
    {
        "type": "function",
        "function": {
            "name": "sales_agent",
            "description": (
                "Delegate order fulfillment decisions and sales "
                "processing to the Sales Agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string"
                    },
                    "quantity": {
                        "type": "integer"
                    },
                    "total_price": {
                        "type": "number"
                    },
                    "request_date": {
                        "type": "string"
                    },
                    "inventory_result": {
                        "type": "object"
                    }
                },
                "required": [
                    "item_name",
                    "quantity",
                    "total_price",
                    "request_date",
                    "inventory_result"
                ]
            }
        }
    }
]

# Mapping
ORCHESTRATOR_TOOL_FUNCTIONS = {
    "inventory_agent": delegate_to_inventory_agent,
    "quote_agent": delegate_to_quote_agent,
    "sales_agent": delegate_to_sales_agent
}

# Orchestrator Agent
 
def orchestrator_agent(
    customer_request: str,
    item_name: str,
    quantity: int,
    request_date: str,
    total_price: float = 0.0
) -> dict:
    """
    LLM-powered Orchestrator Agent.
 
    The LLM decides which specialized agent to delegate to
    and when based on the customer request and agent results.
    """
 
    system_prompt = """
You are the Orchestrator Agent for a paper supply company.
 
Your responsibility is to understand the customer's request
and coordinate the appropriate specialized agents.
 
You have access to three specialized agents as tools:
 
1. inventory_agent
   Use this agent when inventory availability or supplier
   delivery information is required.
 
2. quote_agent
   Use this agent when quote preparation or historical quote
   information is required.
 
3. sales_agent
   Use this agent when an order may be ready for fulfillment.
 
IMPORTANT:
 
You must decide which specialized agent to call and when.
 
Do NOT automatically call Inventory, Quote, and Sales in a
hard-coded sequence.
 
Use the results returned by one agent to decide whether another
agent needs to be called.
 
When calling another agent, pass relevant results from previously
called agents to that agent.
 
Do not perform inventory checks yourself.
Do not search quote history yourself.
Do not process sales yourself.
 
Delegate those responsibilities to the appropriate specialized
agent.
 
Do not invent tool or agent results.
 
When the customer's request has been completely handled,
return the final response as valid JSON only.
"""
 
    user_request = f"""
Customer request: {customer_request}
Item name: {item_name}
Quantity: {quantity}
Request date: {request_date}
Total price: {total_price}
 
Determine which specialized agents are required to handle
this customer request and coordinate them appropriately.
"""
 
    return run_llm_agent(
    system_prompt=system_prompt,
    user_request=user_request,
    tools=ORCHESTRATOR_TOOLS,
    tool_functions=ORCHESTRATOR_TOOL_FUNCTIONS
)
 

def call_your_multi_agent_system(request_with_date: str) -> dict:
    """
    Main entry point for the multi-agent system.
 
    Accepts a natural-language customer request containing
    the request date and routes it through the appropriate agents.
    """
 
    try:
        print("\n=== MULTI-AGENT SYSTEM ===")
        print(f"Customer request: {request_with_date}")
 
        # Normalize request
        request_text = str(request_with_date).strip()
 
        if not request_text:
            return {
                "success": False,
                "status": "invalid_request",
                "response": "Customer request is empty."
            }
 
        # Extract quantity from request
        import re
 
        quantity_match = re.search(
            r"\b(\d+)\b",
            request_text
        )
 
        quantity = (
            int(quantity_match.group(1))
            if quantity_match
            else 1
        )
 
        # Identify inventory item
        inventory = get_all_inventory(
            as_of_date=datetime.now().isoformat()
        )
 
        item_name = None
 
        for inventory_item in inventory.keys():
            if inventory_item.lower() in request_text.lower():
                item_name = inventory_item
                break
 
        # If exact inventory name was not found
        if item_name is None:
            return {
                "success": False,
                "status": "item_not_found",
                "response": (
                    "Unable to identify the requested paper "
                    "type from the customer request."
                )
            }
 
        # Extract request date
        date_match = re.search(
            r"\d{4}-\d{2}-\d{2}",
            request_text
        )
 
        if date_match:
            request_date = date_match.group(0)
        else:
            request_date = datetime.now().strftime("%Y-%m-%d")
 
        # Call orchestrator
        result = orchestrator_agent(
            customer_request=request_text,
            item_name=item_name,
            quantity=quantity,
            request_date=request_date,
            total_price=0.0
        )
 
        return {
            "success": True,
            "status": "completed",
            "response": result
        }
 
    except Exception as e:
 
        print(f"ERROR in multi-agent system: {e}")
 
        return {
            "success": False,
            "status": "error",
            "response": str(e)
        }

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    """
    Run end-to-end scenarios through the LLM-powered orchestrator.
    """
 
    scenarios = [
        {
            "name": "Available inventory",
            "customer_request": (
                "I need a quote for 10 units of A4 paper. "
                "If enough inventory is available, process the order."
            ),
            "item_name": "A4 paper",
            "quantity": 10,
            "request_date": "2026-09-18",
            "total_price": 100.00
        },
        {
            "name": "Insufficient inventory",
            "customer_request": (
                "I need 10000 units of A4 paper. "
                "Check availability and provide a quote. "
                "Process the order only if enough inventory "
                "is currently available."
            ),
            "item_name": "A4 paper",
            "quantity": 10000,
            "request_date": "2026-09-18",
            "total_price": 100000.00
        }
    ]
 
    results = []
 
    for scenario in scenarios:
        print("\n" + "=" * 60)
        print(f"SCENARIO: {scenario['name']}")
        print("=" * 60)
 
        result = orchestrator_agent(
            customer_request=scenario["customer_request"],
            item_name=scenario["item_name"],
            quantity=scenario["quantity"],
            request_date=scenario["request_date"],
            total_price=scenario["total_price"]
        )
 
        results.append(result)
 
        print("\n--- FINAL ORCHESTRATOR RESULT ---")
        print(json.dumps(result, indent=2, default=str))
 
    return results
if __name__ == "__main__":
    test_result = orchestrator_agent(
        customer_request=(
            "I need 10000 units of A4 paper. "
            "Check availability and provide a quote. "
            "Process the order only if enough inventory is currently available."
        ),
        item_name="A4 paper",
        quantity=10000,
        request_date="2026-09-18",
        total_price=100000.00
    )
 
    print("\n--- ORCHESTRATOR INSUFFICIENT INVENTORY TEST ---")
    print(json.dumps(test_result, indent=2, default=str))