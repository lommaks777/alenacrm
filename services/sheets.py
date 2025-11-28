import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import gspread_asyncio
from google.oauth2.service_account import Credentials
from config import Config

logger = logging.getLogger(__name__)


class SheetsService:
    """Async Google Sheets service with locking for concurrent access"""
    
    # Worksheet names
    INVENTORY_SHEET = "Inventory"
    CLIENTS_SHEET = "Clients"
    TRANSACTIONS_SHEET = "Transactions"
    PREORDERS_SHEET = "Preorders"
    
    def __init__(self):
        self.agcm = None
        self.spreadsheet = None
        self.lock = asyncio.Lock()
        self._initialized = False
    
    def _get_creds(self):
        """Get Google credentials"""
        creds_dict = Config.get_google_credentials()
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        return Credentials.from_service_account_info(creds_dict, scopes=scopes)
    
    async def initialize(self):
        """Initialize the Google Sheets connection"""
        if self._initialized:
            return
        
        try:
            self.agcm = gspread_asyncio.AsyncioGspreadClientManager(self._get_creds)
            agc = await self.agcm.authorize()
            self.spreadsheet = await agc.open_by_key(Config.GOOGLE_SHEET_KEY)
            
            # Verify worksheets exist
            await self._verify_worksheets()
            
            self._initialized = True
            logger.info("Google Sheets service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets: {e}")
            raise
    
    async def _verify_worksheets(self):
        """Verify that required worksheets exist"""
        worksheets = await self.spreadsheet.worksheets()
        worksheet_names = [ws.title for ws in worksheets]
        
        required = [self.INVENTORY_SHEET, self.CLIENTS_SHEET, self.TRANSACTIONS_SHEET, self.PREORDERS_SHEET]
        missing = [name for name in required if name not in worksheet_names]
        
        if missing:
            logger.warning(f"Missing worksheets: {missing}. Creating them...")
            for name in missing:
                await self.spreadsheet.add_worksheet(title=name, rows=1000, cols=10)
                await self._initialize_worksheet_headers(name)
    
    async def _initialize_worksheet_headers(self, worksheet_name: str):
        """Initialize headers for a worksheet"""
        worksheet = await self.spreadsheet.worksheet(worksheet_name)
        
        headers = {
            self.INVENTORY_SHEET: ["SKU", "Name", "Size", "Qty", "Price", "Purchase_Price", "Last_Updated"],
            self.CLIENTS_SHEET: ["Name", "Instagram", "Telegram", "Description", "Transactions", "Reminder_Date", "Reminder_Text"],
            self.TRANSACTIONS_SHEET: ["Timestamp", "Type", "Client_Name", "Item_Name", "Size", "Price", "Qty", "Total_Amount"],
            self.PREORDERS_SHEET: ["Timestamp", "Client_Name", "Item_Name", "Quantity", "Description", "Notes", "Status"]
        }
        
        if worksheet_name in headers:
            await worksheet.update('A1', [headers[worksheet_name]])
            logger.info(f"Initialized headers for {worksheet_name}")
    
    async def get_all_products(self) -> List[str]:
        """Get all unique product names from inventory (for fuzzy matching)"""
        try:
            worksheet = await self.spreadsheet.worksheet(self.INVENTORY_SHEET)
            records = await worksheet.get_all_records()
            
            # Extract unique product names
            products = list(set(record.get('Name', '') for record in records if record.get('Name')))
            logger.info(f"Retrieved {len(products)} unique products from inventory")
            return products
        except Exception as e:
            logger.error(f"Failed to get products: {e}")
            return []
    
    async def update_inventory(self, items: List[Dict[str, Any]], transaction_type: str) -> List[Dict[str, Any]]:
        """
        Update inventory and log transactions
        
        Args:
            items: List of items with keys: name, size, quantity, price (optional)
            transaction_type: "Supply" or "Sale"
        
        Returns:
            List of updated items with current quantities
            
        Raises:
            ValueError: If sale item not found in inventory or insufficient stock
        """
        
        def normalize_size(size: str) -> str:
            """Normalize size - convert Russian letters to English"""
            size = size.upper().strip()
            # Convert Russian letters to English equivalents
            replacements = {
                'М': 'M',  # Russian M
                'Л': 'L',  # Russian L  
                'С': 'S',  # Russian S
                'Х': 'X',  # Russian X
            }
            for ru, en in replacements.items():
                size = size.replace(ru, en)
            return size
        
        def names_match(name1: str, name2: str) -> bool:
            """Simple case-insensitive, whitespace-normalized comparison"""
            clean1 = ' '.join(name1.strip().lower().split())
            clean2 = ' '.join(name2.strip().lower().split())
            return clean1 == clean2
        
        async with self.lock:
            try:
                inventory_ws = await self.spreadsheet.worksheet(self.INVENTORY_SHEET)
                transactions_ws = await self.spreadsheet.worksheet(self.TRANSACTIONS_SHEET)
                
                # Check and fix Inventory sheet headers
                inv_values = await inventory_ws.get_all_values()
                expected_inv_headers = ["SKU", "Name", "Size", "Qty", "Price", "Purchase_Price", "Last_Updated"]
                
                if not inv_values or inv_values[0] != expected_inv_headers:
                    logger.warning("Inventory sheet headers are incorrect, fixing...")
                    # Insert headers at row 1, shifting existing data down
                    await inventory_ws.insert_row(expected_inv_headers, index=1)
                    logger.info("Fixed Inventory sheet headers")
                
                # Check and fix Transactions sheet headers
                trans_values = await transactions_ws.get_all_values()
                expected_trans_headers = ["Timestamp", "Type", "Client_Name", "Item_Name", "Size", "Price", "Qty", "Total_Amount"]
                
                if not trans_values or trans_values[0] != expected_trans_headers:
                    logger.warning("Transactions sheet headers are incorrect, fixing...")
                    # Insert headers at row 1, shifting existing data down
                    await transactions_ws.insert_row(expected_trans_headers, index=1)
                    logger.info("Fixed Transactions sheet headers")
                
                records = await inventory_ws.get_all_records()
                timestamp = datetime.now().isoformat()
                
                logger.info(f"Loaded {len(records)} inventory records")
                if records:
                    logger.info(f"Sample record: {records[0]}")
                
                # For Sales: validate stock availability BEFORE making any changes
                if transaction_type == "Sale":
                    for item in items:
                        name = item['name']
                        size = item['size']
                        qty_needed = item['quantity']
                        
                        # Find the item in inventory using fuzzy matching
                        found = False
                        matched_name = None
                        for record in records:
                            if names_match(record.get('Name', ''), name) and \
                               normalize_size(record.get('Size', '')) == normalize_size(size):
                                found = True
                                matched_name = record.get('Name', '')
                                current_qty = int(record.get('Qty', 0))
                                
                                if current_qty < qty_needed:
                                    raise ValueError(
                                        f"Недостаточно товара '{matched_name}' размер {size}. "
                                        f"На складе: {current_qty} шт, запрошено: {qty_needed} шт"
                                    )
                                break
                        
                        if not found:
                            # List available products for helpful error message
                            available = set()
                            for record in records:
                                rec_name = record.get('Name', '')
                                rec_size = record.get('Size', '')
                                if rec_name:
                                    available.add(f"{rec_name} ({rec_size})")
                            
                            available_list = ", ".join(sorted(available)[:5])  # Show first 5
                            raise ValueError(
                                f"Товар '{name}' размер {size} не найден на складе.\n\n"
                                f"Доступные товары: {available_list}"
                            )
                
                updated_items = []
                
                for item in items:
                    name = item['name']
                    size = item['size']
                    qty_change = item['quantity']
                    price = item.get('price', 0)  # Sale price
                    purchase_price = item.get('purchase_price', 0)  # Purchase price (закупочная цена)
                    
                    # Find existing row using fuzzy matching
                    existing_row = None
                    row_index = None
                    
                    for idx, record in enumerate(records):
                        if names_match(record.get('Name', ''), name) and \
                           normalize_size(record.get('Size', '')) == normalize_size(size):
                            existing_row = record
                            row_index = idx + 2  # +2 for header and 1-based indexing
                            # Use the exact name from inventory
                            name = record.get('Name', '')
                            break
                    
                    if existing_row:
                        # Update existing product
                        current_qty = int(existing_row.get('Qty', 0))
                        
                        if transaction_type == "Supply":
                            new_qty = current_qty + qty_change
                            # For Supply: update prices if provided
                            update_price = price if price > 0 else existing_row.get('Price', 0)
                            update_purchase_price = purchase_price if purchase_price > 0 else existing_row.get('Purchase_Price', 0)
                        else:  # Sale
                            new_qty = current_qty - qty_change
                            # For Sale: NEVER update prices (keep existing prices)
                            update_price = existing_row.get('Price', 0)
                            update_purchase_price = existing_row.get('Purchase_Price', 0)
                        
                        # Update the row
                        sku = existing_row.get('SKU', f"{name}_{size}")
                        
                        await inventory_ws.update(f'A{row_index}:G{row_index}', [[
                            sku, name, size, new_qty, update_price, update_purchase_price, timestamp
                        ]])
                        
                        updated_items.append({
                            'name': name,
                            'size': size,
                            'qty': new_qty,
                            'price': update_price
                        })
                    else:
                        # Only add new product for Supply (already validated for Sale above)
                        if transaction_type == "Supply":
                            sku = f"{name}_{size}"
                            new_qty = qty_change
                            
                            await inventory_ws.append_row([
                                sku, name, size, new_qty, price, purchase_price, timestamp
                            ])
                            
                            updated_items.append({
                                'name': name,
                                'size': size,
                                'qty': new_qty,
                                'price': price
                            })
                    
                    # Log transaction
                    client_name = item.get('client_name', '')
                    total_amount = price * qty_change
                    
                    await transactions_ws.append_row([
                        timestamp, transaction_type, client_name, name, size, price, qty_change, total_amount
                    ])
                
                logger.info(f"Updated inventory with {len(updated_items)} items ({transaction_type})")
                return updated_items
                
            except ValueError:
                # Re-raise validation errors
                raise
            except Exception as e:
                logger.error(f"Failed to update inventory: {e}")
                raise
    
    async def get_stock_by_name(self, product_name: str) -> List[Dict[str, Any]]:
        """Get stock levels for all sizes of a product"""
        try:
            worksheet = await self.spreadsheet.worksheet(self.INVENTORY_SHEET)
            records = await worksheet.get_all_records()
            
            stock_levels = []
            for record in records:
                if record.get('Name', '').lower() == product_name.lower():
                    stock_levels.append({
                        'size': record.get('Size', ''),
                        'qty': int(record.get('Qty', 0))
                    })
            
            return stock_levels
        except Exception as e:
            logger.error(f"Failed to get stock levels: {e}")
            return []
    
    async def get_client(self, client_name: str) -> Optional[Dict[str, Any]]:
        """Get client information by name"""
        try:
            worksheet = await self.spreadsheet.worksheet(self.CLIENTS_SHEET)
            
            # Define expected headers
            expected_headers = ["Name", "Instagram", "Telegram", "Description", "Transactions", "Reminder_Date", "Reminder_Text"]
            
            try:
                records = await worksheet.get_all_records(expected_headers=expected_headers)
            except Exception as e:
                logger.warning(f"Issue with client sheet: {e}")
                return None
            
            # Find client by name
            for record in records:
                if record.get('Name', '').lower() == client_name.lower():
                    return {
                        'name': record.get('Name', ''),
                        'instagram': record.get('Instagram', ''),
                        'telegram': record.get('Telegram', ''),
                        'description': record.get('Description', ''),
                        'transactions': record.get('Transactions', ''),
                        'reminder_date': record.get('Reminder_Date', ''),
                        'reminder_text': record.get('Reminder_Text', '')
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get client: {e}")
            return None
    
    async def upsert_client(self, client_data: Dict[str, Any]) -> None:
        """Update or insert client data"""
        async with self.lock:
            try:
                worksheet = await self.spreadsheet.worksheet(self.CLIENTS_SHEET)
                
                # Define expected headers with new structure
                expected_headers = ["Name", "Instagram", "Telegram", "Description", "Transactions", "Reminder_Date", "Reminder_Text"]
                
                # Get all values to check headers and data
                all_values = await worksheet.get_all_values()
                
                # Check if headers exist and are correct
                if not all_values or all_values[0] != expected_headers:
                    # Fix headers
                    await worksheet.update('A1:G1', [expected_headers])
                    logger.info("Fixed Clients sheet headers to new structure")
                    all_values = await worksheet.get_all_values()
                
                # Parse records manually from all_values
                records = []
                if len(all_values) > 1:  # If there's data beyond headers
                    headers = all_values[0]
                    for row_data in all_values[1:]:
                        # Pad row with empty strings if needed
                        while len(row_data) < len(headers):
                            row_data.append('')
                        record = dict(zip(headers, row_data))
                        records.append(record)
                
                name = client_data['name']
                instagram = client_data.get('instagram', '')
                telegram = client_data.get('telegram', '')
                description = client_data.get('description', '')  # New: for client characteristics
                transaction = client_data.get('transaction', '')  # New: for purchase history
                reminder_date = client_data.get('reminder_date', '')
                reminder_text = client_data.get('reminder_text', '')
                
                # Find existing client (case-insensitive and strip whitespace)
                existing_row = None
                row_index = None
                
                for idx, record in enumerate(records):
                    record_name = record.get('Name', '').strip()
                    if record_name.lower() == name.strip().lower():
                        existing_row = record
                        row_index = idx + 2  # +2 for header row and 1-based indexing
                        logger.info(f"Found existing client '{record_name}' at row {row_index}")
                        break
                
                if existing_row:
                    # Update existing client
                    # Append to description if provided
                    current_description = existing_row.get('Description', '').strip()
                    if description:
                        if current_description:
                            updated_description = f"{current_description}\n{description}"
                        else:
                            updated_description = description
                    else:
                        updated_description = current_description
                    
                    # Append to transactions if provided
                    current_transactions = existing_row.get('Transactions', '').strip()
                    if transaction:
                        if current_transactions:
                            updated_transactions = f"{current_transactions}\n{transaction}"
                        else:
                            updated_transactions = transaction
                    else:
                        updated_transactions = current_transactions
                    
                    # Update contact info if provided
                    updated_instagram = instagram or existing_row.get('Instagram', '')
                    updated_telegram = telegram or existing_row.get('Telegram', '')
                    
                    # Update reminder if provided (replace, not append)
                    updated_reminder_date = reminder_date or existing_row.get('Reminder_Date', '')
                    updated_reminder_text = reminder_text or existing_row.get('Reminder_Text', '')
                    
                    await worksheet.update(f'A{row_index}:G{row_index}', [[
                        name, updated_instagram, updated_telegram, updated_description,
                        updated_transactions, updated_reminder_date, updated_reminder_text
                    ]])
                    logger.info(f"Updated existing client: {name}")
                else:
                    # Add new client
                    await worksheet.append_row([
                        name, instagram, telegram, description, transaction, reminder_date, reminder_text
                    ])
                    logger.info(f"Added new client: {name}")
                
            except Exception as e:
                logger.error(f"Failed to upsert client: {e}")
                raise
    
    async def get_reminders_for_today(self) -> List[Dict[str, str]]:
        """Get all clients with reminders scheduled for today"""
        try:
            worksheet = await self.spreadsheet.worksheet(self.CLIENTS_SHEET)
            records = await worksheet.get_all_records()
            
            today = datetime.now().strftime('%Y-%m-%d')
            reminders = []
            
            for record in records:
                reminder_date = record.get('Reminder_Date', '')
                if reminder_date == today:
                    reminders.append({
                        'name': record.get('Name', ''),
                        'text': record.get('Reminder_Text', '')
                    })
            
            logger.info(f"Found {len(reminders)} reminders for today")
            return reminders
            
        except Exception as e:
            logger.error(f"Failed to get reminders: {e}")
            return []
    
    async def clear_reminder(self, client_name: str) -> None:
        """Clear reminder for a specific client"""
        async with self.lock:
            try:
                worksheet = await self.spreadsheet.worksheet(self.CLIENTS_SHEET)
                records = await worksheet.get_all_records()
                
                for idx, record in enumerate(records):
                    if record.get('Name', '').lower() == client_name.lower():
                        row_index = idx + 2
                        # Clear reminder columns (E and F)
                        await worksheet.update(f'E{row_index}:F{row_index}', [['', '']])
                        logger.info(f"Cleared reminder for client: {client_name}")
                        break
                        
            except Exception as e:
                logger.error(f"Failed to clear reminder: {e}")
                raise
    
    async def undo_sale(self, client_name: str, item_name: str, size: str, quantity: int, timestamp: str) -> bool:
        """
        Undo a sale transaction
        
        Returns True if successful, False otherwise
        """
        async with self.lock:
            try:
                # Reverse inventory change
                inventory_ws = await self.spreadsheet.worksheet(self.INVENTORY_SHEET)
                transactions_ws = await self.spreadsheet.worksheet(self.TRANSACTIONS_SHEET)
                clients_ws = await self.spreadsheet.worksheet(self.CLIENTS_SHEET)
                
                # Add quantity back to inventory
                inv_records = await inventory_ws.get_all_records()
                for idx, record in enumerate(inv_records):
                    if record.get('Name', '').lower() == item_name.lower() and \
                       record.get('Size', '').lower() == size.lower():
                        row_index = idx + 2
                        current_qty = int(record.get('Qty', 0))
                        new_qty = current_qty + quantity
                        
                        await inventory_ws.update(f'D{row_index}', [[new_qty]])
                        break
                
                # Remove transaction from Transactions sheet
                trans_records = await transactions_ws.get_all_records()
                for idx, record in enumerate(trans_records):
                    if (record.get('Timestamp', '') == timestamp and
                        record.get('Type', '') == 'Sale' and
                        record.get('Client_Name', '') == client_name and
                        record.get('Item_Name', '').lower() == item_name.lower() and
                        record.get('Size', '').lower() == size.lower()):
                        row_index = idx + 2
                        await transactions_ws.delete_rows(row_index)
                        break
                
                logger.info(f"Undone sale: {item_name} {size} for {client_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to undo sale: {e}")
                return False


    async def undo_last_sale(self) -> bool:
        """
        Undo the most recent sale transaction(s)
        Now supports undoing multiple items from a single sale (same timestamp)
        
        Returns True if successful, False otherwise
        """
        async with self.lock:
            try:
                inventory_ws = await self.spreadsheet.worksheet(self.INVENTORY_SHEET)
                transactions_ws = await self.spreadsheet.worksheet(self.TRANSACTIONS_SHEET)
                
                # Get all transactions using get_all_values
                trans_values = await transactions_ws.get_all_values()
                
                if not trans_values or len(trans_values) < 2:
                    logger.warning("No transactions found")
                    return False
                
                # Parse transactions manually
                headers = trans_values[0]
                logger.info(f"Transaction headers: {headers}")
                trans_records = []
                for row_data in trans_values[1:]:
                    while len(row_data) < len(headers):
                        row_data.append('')
                    record = dict(zip(headers, row_data))
                    trans_records.append(record)
                
                logger.info(f"Found {len(trans_records)} transactions")
                if trans_records:
                    logger.info(f"Last transaction: {trans_records[-1]}")
                
                # Find the last Sale transaction and its timestamp
                last_sale_timestamp = None
                
                for idx in range(len(trans_records) - 1, -1, -1):
                    record_type = trans_records[idx].get('Type', '')
                    logger.info(f"Checking transaction {idx}: Type='{record_type}'")
                    if record_type == 'Sale':
                        last_sale_timestamp = trans_records[idx].get('Timestamp', '')
                        logger.info(f"Found last sale with timestamp: {last_sale_timestamp}")
                        break
                
                if not last_sale_timestamp:
                    logger.warning("No sale transaction found to undo")
                    return False
                
                # Find ALL sales with this timestamp (for multi-item sales)
                sales_to_undo = []
                for idx, record in enumerate(trans_records):
                    if record.get('Type', '') == 'Sale' and record.get('Timestamp', '') == last_sale_timestamp:
                        sales_to_undo.append({
                            'record': record,
                            'row_index': idx + 2  # +2 for header and 1-based indexing
                        })
                
                logger.info(f"Found {len(sales_to_undo)} items to undo from last sale")
                
                # Restore inventory for each item
                inv_values = await inventory_ws.get_all_values()
                if not inv_values or len(inv_values) < 2:
                    logger.error("No inventory data found")
                    return False
                
                inv_headers = inv_values[0]
                
                for sale in sales_to_undo:
                    item_name = sale['record'].get('Item_Name', '')
                    size = sale['record'].get('Size', '')
                    quantity = int(sale['record'].get('Qty', 0))
                    
                    # Find and update inventory
                    for idx, row_data in enumerate(inv_values[1:], start=2):
                        while len(row_data) < len(inv_headers):
                            row_data.append('')
                        record = dict(zip(inv_headers, row_data))
                        
                        if record.get('Name', '').lower() == item_name.lower() and \
                           record.get('Size', '').lower() == size.lower():
                            current_qty = int(record.get('Qty', 0))
                            new_qty = current_qty + quantity
                            
                            await inventory_ws.update(f'D{idx}', [[new_qty]])
                            logger.info(f"Restored {quantity} units of {item_name} {size}")
                            break
                
                # Delete all transaction rows (starting from the last to avoid index shifting)
                for sale in reversed(sales_to_undo):
                    await transactions_ws.delete_rows(sale['row_index'])
                
                logger.info(f"Undone last sale: {len(sales_to_undo)} items")
                return True
                
            except Exception as e:
                logger.error(f"Failed to undo last sale: {e}")
                return False


    async def undo_last_supply(self) -> bool:
        """
        Undo the most recent supply transaction
        
        Returns True if successful, False otherwise
        """
        async with self.lock:
            try:
                inventory_ws = await self.spreadsheet.worksheet(self.INVENTORY_SHEET)
                transactions_ws = await self.spreadsheet.worksheet(self.TRANSACTIONS_SHEET)
                
                # Get all transactions using get_all_values
                trans_values = await transactions_ws.get_all_values()
                
                if not trans_values or len(trans_values) < 2:
                    logger.warning("No transactions found")
                    return False
                
                # Parse transactions manually
                headers = trans_values[0]
                trans_records = []
                for row_data in trans_values[1:]:
                    while len(row_data) < len(headers):
                        row_data.append('')
                    record = dict(zip(headers, row_data))
                    trans_records.append(record)
                
                # Find the last Supply transaction
                last_supply = None
                last_supply_index = None
                
                for idx in range(len(trans_records) - 1, -1, -1):
                    if trans_records[idx].get('Type', '') == 'Supply':
                        last_supply = trans_records[idx]
                        last_supply_index = idx + 2  # +2 for header and 1-based indexing
                        break
                
                if not last_supply:
                    logger.warning("No supply transaction found to undo")
                    return False
                
                # Extract supply details
                item_name = last_supply.get('Item_Name', '')
                size = last_supply.get('Size', '')
                quantity = int(last_supply.get('Qty', 0))
                
                # Subtract quantity from inventory using get_all_values
                inv_values = await inventory_ws.get_all_values()
                if not inv_values or len(inv_values) < 2:
                    logger.error("No inventory data found")
                    return False
                
                inv_headers = inv_values[0]
                for idx, row_data in enumerate(inv_values[1:], start=2):
                    while len(row_data) < len(inv_headers):
                        row_data.append('')
                    record = dict(zip(inv_headers, row_data))
                    
                    if record.get('Name', '').lower() == item_name.lower() and \
                       record.get('Size', '').lower() == size.lower():
                        current_qty = int(record.get('Qty', 0))
                        new_qty = max(0, current_qty - quantity)  # Don't go below 0
                        
                        await inventory_ws.update(f'D{idx}', [[new_qty]])
                        break
                
                # Delete the transaction row
                await transactions_ws.delete_rows(last_supply_index)
                
                logger.info(f"Undone last supply: {item_name} {size}, quantity: {quantity}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to undo last supply: {e}")
                return False


    async def undo_last_client_update(self, client_name: str) -> bool:
        """
        Remove the last line from client's Description field
        
        Returns True if successful, False otherwise
        """
        async with self.lock:
            try:
                worksheet = await self.spreadsheet.worksheet(self.CLIENTS_SHEET)
                
                # Define expected headers
                expected_headers = ["Name", "Instagram", "Telegram", "Description", "Transactions", "Reminder_Date", "Reminder_Text"]
                
                # Get all values
                all_values = await worksheet.get_all_values()
                
                if not all_values or len(all_values) < 2:
                    logger.warning("No client data found")
                    return False
                
                # Parse records
                headers = all_values[0]
                for idx, row_data in enumerate(all_values[1:], start=2):
                    while len(row_data) < len(headers):
                        row_data.append('')
                    record = dict(zip(headers, row_data))
                    
                    if record.get('Name', '').strip().lower() == client_name.strip().lower():
                        # Found the client
                        current_description = record.get('Description', '').strip()
                        
                        if not current_description:
                            logger.warning(f"No description to undo for client: {client_name}")
                            return False
                        
                        # Remove last line from description
                        lines = current_description.split('\n')
                        if len(lines) > 1:
                            updated_description = '\n'.join(lines[:-1])
                        else:
                            updated_description = ''  # Remove all if only one line
                        
                        # Update the description column (column D, index 3)
                        await worksheet.update(f'D{idx}', [[updated_description]])
                        
                        logger.info(f"Undone last client update for: {client_name}")
                        return True
                
                logger.warning(f"Client not found: {client_name}")
                return False
                
            except Exception as e:
                logger.error(f"Failed to undo client update: {e}")
                return False
    
    async def add_preorder(self, client_name: str, items: List[Dict[str, Any]], notes: Optional[str] = None) -> bool:
        """
        Add preorder to Preorders sheet
        
        Args:
            client_name: Client's full name
            items: List of items with keys: item_name, quantity, description (optional)
            notes: General notes about the preorder
            
        Returns:
            True if successful, False otherwise
        """
        async with self.lock:
            try:
                preorders_ws = await self.spreadsheet.worksheet(self.PREORDERS_SHEET)
                timestamp = datetime.now().isoformat()
                
                # Add each item as a separate row
                rows_to_add = []
                for item in items:
                    row = [
                        timestamp,
                        client_name,
                        item['item_name'],
                        item['quantity'],
                        item.get('description', ''),
                        notes or '',
                        'Новый'  # Status: New
                    ]
                    rows_to_add.append(row)
                
                # Append all rows
                await preorders_ws.append_rows(rows_to_add)
                
                logger.info(f"Added preorder for {client_name}: {len(items)} items")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add preorder: {e}")
                return False


# Global instance
sheets_service = SheetsService()
