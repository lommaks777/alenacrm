import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from config import Config

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)





class ClientEditData(BaseModel):
    """Client information edit data"""
    client_name: str = Field(description="Client full name")
    notes: str = Field(description="Notes or additional information to add about the client")


# Pydantic Models for Structured Outputs

class SupplyItem(BaseModel):
    """Single item in a supply restock"""
    name: str = Field(description="Product name, matched to existing inventory")
    size: str = Field(description="Size designation (e.g., S, M, L, XL)")
    quantity: int = Field(description="Quantity being restocked, must be greater than 0", gt=0)
    price: float = Field(default=0, description="Unit sale price for this item (optional, defaults to 0 if not specified)")
    purchase_price: float = Field(default=0, description="Unit purchase price (закупочная цена) for this item (optional, defaults to 0 if not specified)")


class SupplyData(BaseModel):
    """Complete supply transaction data"""
    items: List[SupplyItem] = Field(description="List of items being restocked")


class ClientInfo(BaseModel):
    """Customer information"""
    name: str = Field(description="Full client name")
    instagram: Optional[str] = Field(default=None, description="Instagram handle")
    telegram: Optional[str] = Field(default=None, description="Telegram username or ID")
    notes: Optional[str] = Field(default=None, description="Additional notes about the client")


class SaleInfo(BaseModel):
    """Sale transaction details for a single item"""
    item_name: str = Field(description="Product name being sold")
    size: str = Field(description="Product size")
    quantity: int = Field(description="Quantity sold", gt=0)
    price: float = Field(description="Unit price for this item", gt=0)


class ReminderInfo(BaseModel):
    """Reminder scheduling information"""
    days_from_now: int = Field(description="Number of days from today to set the reminder")
    text: str = Field(description="Reminder message text")


class SaleData(BaseModel):
    """Complete sale transaction data - supports multiple items"""
    client: ClientInfo
    items: List[SaleInfo] = Field(description="List of items being sold in this transaction")
    reminder: Optional[ReminderInfo] = Field(default=None, description="Optional reminder to schedule")


class PreorderItem(BaseModel):
    """Single item in a preorder"""
    item_name: str = Field(description="Product name")
    quantity: int = Field(description="Quantity ordered", gt=0)
    description: Optional[str] = Field(default=None, description="Additional description or notes about the item")


class PreorderData(BaseModel):
    """Complete preorder data"""
    client_name: str = Field(description="Client's full name")
    items: List[PreorderItem] = Field(description="List of items in the preorder")
    notes: Optional[str] = Field(default=None, description="General notes about the preorder")


class AIService:
    """OpenAI integration for transcription and NLP"""
    
    @staticmethod
    async def transcribe_audio(audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio file using Whisper API
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            with open(audio_file_path, 'rb') as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="json"
                )
            
            text = transcript.text
            logger.info(f"Transcription successful: {text[:100]}...")
            return text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    @staticmethod
    async def classify_message(text: str) -> str:
        """
        Classify if message is about Supply, Sale, Client Edit, or Query
        
        Returns:
            "supply", "sale", "client_edit", or "query"
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a warehouse assistant. Determine if the user's message is about:
- SUPPLY: Restocking inventory, receiving new products, incoming stock, adding items
- SALE: Customer purchase, selling products, client transaction (includes mentions of price, buying, purchasing)
- PREORDER: Customer wants to order items for future delivery, "предзаказ", "заказать", "хочет заказать"
- CLIENT_EDIT: ONLY adding personal notes/characteristics about client WITHOUT any sale/purchase information
- QUERY: Questions about inventory, stock levels, asking "how many", "what's in stock", "show me"

Key indicators:
- SALE: mentions price, buying, purchasing, "купила", "купил", "за X долларов", PAST TENSE purchases
- PREORDER: "предзаказ", "заказать", "хочет заказать", FUTURE orders without immediate payment
- CLIENT_EDIT: ONLY preferences, interests, characteristics WITHOUT purchase details
- SUPPLY: adding to stock, "добавь", "поставка", receiving products, "пришло"
- QUERY: questions about stock, "сколько", "что на складе", "покажи", "есть ли"

Respond with only one word: "supply", "sale", "preorder", "client_edit", or "query"."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0
            )
            
            classification = response.choices[0].message.content.strip().lower()
            logger.info(f"Message classified as: {classification}")
            return classification if classification in ["supply", "sale", "preorder", "client_edit", "query"] else "supply"
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "supply"  # Default to supply
    
    @staticmethod
    async def parse_supply(text: str, existing_products: List[str]) -> Optional[SupplyData]:
        """
        Parse supply/restock information from transcribed text
        
        Args:
            text: Transcribed text
            existing_products: List of existing product names for semantic matching
            
        Returns:
            SupplyData object or None if parsing failed
        """
        try:
            # Create product list context
            products_context = "\n".join([f"- {p}" for p in existing_products]) if existing_products else "No existing products"
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a warehouse assistant extracting product restock information.

AVAILABLE INVENTORY PRODUCTS:
{products_context}

PRODUCT MATCHING RULES:
- Match to existing inventory ONLY if ALL key characteristics match
- Key characteristics include: color, product type, material, AND special features (high waist, low waist, etc.)
- You MAY normalize: grammatical endings ("Бежевый" vs "Бежевые"), singular/plural ("Трусы" vs "Трусики"), synonyms ("Сетка" vs "Сеточка"), prepositions ("с сеткой" → "Сетка")
- You MUST NOT ignore: special features like "высокая талия", "низкая талия", "бразилиана", "стринги", "слип", etc.
- If special feature is mentioned but not in existing products, create NEW product name with that feature
- ONLY include items with quantity GREATER THAN 0. Skip items with 0 quantity.
- Extract SALE price ("продажа по", "цена продажи") if mentioned. If not mentioned, use 0.
- Extract PURCHASE price ("закупка по", "закупочная цена", "купили по", "стоимость закупки") if mentioned. If not mentioned, use 0.

EXAMPLES:
Input: "добавь бежевый топ сеткой M 5 штук, продажа по 25 долларов, закупка по 15 долларов" (inventory has "Бежевые Топ Сетка")
Output: {{items: [{{name: "Бежевые Топ Сетка", size: "M", quantity: 5, price: 25, purchase_price: 15}}]}}
(Match found - same color, type, material, both prices extracted)

Input: "добавь черные трусики сеточкой M 5 штук закупочная цена 20 долларов" (inventory has "Черные Трусы Сетка")
Output: {{items: [{{name: "Черные Трусы Сетка", size: "M", quantity: 5, price: 0, purchase_price: 20}}]}}
(Match found - normalized "трусики"→"трусы", "сеточкой"→"сетка", only purchase price mentioned)

Input: "добавь черные трусы сетка с высокой талией M 5 штук стоимость закупки 18 долларов" (inventory has "Черные Трусы Сетка")
Output: {{items: [{{name: "Черные Трусы Сетка Высокая Талия", size: "M", quantity: 5, price: 0, purchase_price: 18}}]}}
(NO match - "высокая талия" is a KEY feature not in existing product, so create NEW product with purchase price)

Input: "добавь черные трусы бразилиана M 3 штуки по цене 28 долларов" (inventory has "Черные Трусы Сетка")
Output: {{items: [{{name: "Черные Трусы Бразилиана", size: "M", quantity: 3, price: 28, purchase_price: 0}}]}}
(NO match - "бразилиана" is different style, create NEW product with sale price)

Extract all items being restocked with their names, sizes, quantities, and prices.
Return data in the specified JSON format."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "supply_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "size": {"type": "string"},
                                        "quantity": {"type": "integer"},
                                        "price": {"type": "number"},
                                        "purchase_price": {"type": "number"}
                                    },
                                    "required": ["name", "size", "quantity", "price", "purchase_price"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["items"],
                        "additionalProperties": False
                    }
                }},
                temperature=0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            supply_data = SupplyData(**result)
            
            logger.info(f"Parsed supply data: {len(supply_data.items)} items")
            for item in supply_data.items:
                logger.info(f"  Product matched: '{item.name}'")
            
            return supply_data
            
        except Exception as e:
            logger.error(f"Supply parsing failed: {e}")
            return None
    
    @staticmethod
    async def parse_sale(text: str, current_date: str, existing_products: List[str]) -> Optional[SaleData]:
        """
        Parse sale/customer transaction information from transcribed text
        
        Args:
            text: Transcribed text
            current_date: Current date in YYYY-MM-DD format
            existing_products: List of existing product names for semantic matching
            
        Returns:
            SaleData object or None if parsing failed
        """
        try:
            # Create product list context
            products_context = "\n".join([f"- {p}" for p in existing_products]) if existing_products else "No existing products"
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a CRM assistant extracting customer sale information.

Today's date is: {current_date}

AVAILABLE INVENTORY PRODUCTS:
{products_context}

PRODUCT MATCHING RULES:
- Match to existing inventory ONLY if ALL key characteristics match
- Key characteristics include: color, product type, material, AND special features (high waist, low waist, etc.)
- You MAY normalize: grammatical endings ("Бежевый" vs "Бежевые"), singular/plural ("Трусы" vs "Трусики"), synonyms ("Сетка" vs "Сеточка"), prepositions ("с сеткой" → "Сетка")
- You MUST NOT ignore: special features like "высокая талия", "низкая талия", "бразилиана", "стринги", "слип", etc.
- If special feature is mentioned but not in existing products, create NEW product name with that feature
- Output the EXACT product name from the inventory list if match found, otherwise create new name

Extract information for MULTIPLE items if mentioned:

1. Client information:
   - name: Client's full name
   - instagram: Instagram username ONLY (without "Пользователь Instagram:" prefix)
   - telegram: Telegram username ONLY (without "Пользователь Telegram:" prefix)
   - notes: Personal characteristics, preferences, AND future purchase wishes
   
2. Items (array of ALREADY PURCHASED products):
   - ONLY extract products that were ALREADY BOUGHT/PURCHASED
   - CRITICAL: DO NOT include items client "wants", "will buy", "interested in" - those go to notes
   - Look for past tense: "купила", "купил", "bought", "purchased"
   - For EACH purchased product: name, size, quantity, price
   - Map product names to exact inventory names using matching rules above
   - If price is mentioned once for multiple items, apply it to each item
   - All items must have a price > 0
   
3. Reminder (if mentioned):
   - Convert relative dates to number of days

CRITICAL RULES FOR ITEMS vs NOTES:
- Items array: ONLY products with COMPLETED purchase ("купила", "bought")
- Notes field: Future wishes ("хочет купить", "интересуется", "wants to buy"), preferences, characteristics
- If no price mentioned for purchased items, this is an error - price is REQUIRED for items

EXAMPLES:

Input: "Светлана купила бежевый топ M за 40 долларов" (inventory has "Бежевые Топ Сетка")
Output:
- client: {{
    name: "Светлана",
    instagram: null,
    telegram: null,
    notes: null
  }}
- items: [
    {{item_name: "Бежевые Топ Сетка", size: "M", quantity: 1, price: 40}}
  ]

Input: "Анна купила черные трусики сеточкой M за 35 долларов" (inventory has "Черные Трусы Сетка")
Output:
- client: {{
    name: "Анна",
    notes: null
  }}
- items: [
    {{item_name: "Черные Трусы Сетка", size: "M", quantity: 1, price: 35}}
  ]

Input: "Анастасия купила черные трусы M за 25 долларов. Укажи в описании, что она хочет купить топ бежевый L"
Output:
- client: {{
    name: "Анастасия",
    notes: "Хочет купить топ бежевый L"
  }}
- items: [
    {{item_name: "Черные Трусы Сетка", size: "M", quantity: 1, price: 25}}
  ]
  (Note: future wish goes to notes, not items)

Return data in the specified JSON format."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "sale_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "client": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "instagram": {"type": ["string", "null"]},
                                    "telegram": {"type": ["string", "null"]},
                                    "notes": {"type": ["string", "null"]}
                                },
                                "required": ["name", "instagram", "telegram", "notes"],
                                "additionalProperties": False
                            },
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item_name": {"type": "string"},
                                        "size": {"type": "string"},
                                        "quantity": {"type": "integer"},
                                        "price": {"type": "number"}
                                    },
                                    "required": ["item_name", "size", "quantity", "price"],
                                    "additionalProperties": False
                                }
                            },
                            "reminder": {
                                "type": ["object", "null"],
                                "properties": {
                                    "days_from_now": {"type": "integer"},
                                    "text": {"type": "string"}
                                },
                                "required": ["days_from_now", "text"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["client", "items", "reminder"],
                        "additionalProperties": False
                    }
                }},
                temperature=0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            sale_data = SaleData(**result)
            
            logger.info(f"Parsed sale data: {sale_data.client.name} - {len(sale_data.items)} items")
            for item in sale_data.items:
                logger.info(f"  Product matched: '{item.item_name}'")
            
            return sale_data
            
        except Exception as e:
            logger.error(f"Sale parsing failed: {e}")
            return None
    
    @staticmethod
    async def parse_client_edit(text: str) -> Optional[ClientEditData]:
        """
        Parse client edit information from transcribed text
        
        Args:
            text: Transcribed text
            
        Returns:
            ClientEditData object or None if parsing failed
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a CRM assistant extracting client information updates.

Extract:
1. Client name (the person being discussed)
2. Notes/information to add about the client (preferences, interests, characteristics, etc.)

IMPORTANT:
- The client name should be the person's full name
- Notes should be descriptive information about the client
- Return data in the specified JSON format."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "client_edit_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "client_name": {"type": "string"},
                            "notes": {"type": "string"}
                        },
                        "required": ["client_name", "notes"],
                        "additionalProperties": False
                    }
                }},
                temperature=0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            client_edit_data = ClientEditData(**result)
            
            logger.info(f"Parsed client edit: {client_edit_data.client_name}")
            return client_edit_data
            
        except Exception as e:
            logger.error(f"Client edit parsing failed: {e}")
            return None
    
    @staticmethod
    async def parse_preorder(text: str) -> Optional[PreorderData]:
        """
        Parse preorder information from transcribed text
        
        Args:
            text: Transcribed text
            
        Returns:
            PreorderData object or None if parsing failed
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a warehouse assistant extracting preorder information.

Extract preorder information:
1. Client name (full name)
2. Items (array of items with name, quantity, and optional description)
3. General notes about the preorder (optional)

EXAMPLES:

Input: "Клиент Анна. Предзаказ: черные трусы сетка размер M 2 штуки. Черный топ сетка размер S 1 штука. Забрать в пятницу."
Output:
{
  "client_name": "Анна",
  "items": [
    {"item_name": "Черные Трусы Сетка", "quantity": 2, "description": "размер M"},
    {"item_name": "Черный Топ Сетка", "quantity": 1, "description": "размер S"}
  ],
  "notes": "Забрать в пятницу"
}

Input: "Мария хочет заказать бежевый купальник 3 штуки, размеры разные."
Output:
{
  "client_name": "Мария",
  "items": [
    {"item_name": "Бежевый Купальник", "quantity": 3, "description": "размеры разные"}
  ],
  "notes": null
}

Extract all information and return in JSON format."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "preorder_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "client_name": {"type": "string"},
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item_name": {"type": "string"},
                                        "quantity": {"type": "integer"},
                                        "description": {"type": ["string", "null"]}
                                    },
                                    "required": ["item_name", "quantity", "description"],
                                    "additionalProperties": False
                                }
                            },
                            "notes": {"type": ["string", "null"]}
                        },
                        "required": ["client_name", "items", "notes"],
                        "additionalProperties": False
                    }
                }},
                temperature=0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            preorder_data = PreorderData(**result)
            
            logger.info(f"Parsed preorder: {preorder_data.client_name} - {len(preorder_data.items)} items")
            return preorder_data
            
        except Exception as e:
            logger.error(f"Preorder parsing failed: {e}")
            return None


# Global instance
ai_service = AIService()
