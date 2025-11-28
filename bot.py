import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from config import Config
from services.sheets import sheets_service
from services.ai import ai_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize bot and dispatcher
bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher()

# Scheduler for reminders
scheduler = AsyncIOScheduler(timezone=pytz.timezone(Config.TIMEZONE))


# Authorization middleware
def is_authorized(message: Message) -> bool:
    """Check if user is authorized"""
    return Config.is_user_allowed(message.from_user.id)


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command"""
    if not is_authorized(message):
        logger.warning(f"Unauthorized access attempt from user {message.from_user.id}")
        return
    
    welcome_text = """
🎙️ <b>VoiceStock Bot</b>

Добро пожаловать! Я помогаю управлять складом и клиентами.

<b>Как использовать:</b>
• Отправьте голосовое сообщение о поступлении товара
• Отправьте голосовое сообщение о продаже
• Отправьте голосовое сообщение с заметкой о клиенте
• Используйте команды для управления клиентами
• Я сделаю всё остальное!

<b>Команды:</b>
/client [имя] - посмотреть информацию о клиенте
/edit [имя] | [заметка] - добавить заметку о клиенте

<b>Примеры:</b>
📦 <i>"Получил белые трусики, 5 штук размер M и 3 штуки размер L"</i>
💰 <i>"Клиент Анна Иванова купила черные трусики размер M за 40 долларов, напомни через неделю предложить купальник"</i>
📝 <i>"Добавь информацию о клиенте Анна Иванова: она любит кошек и яркие цвета"</i>
/edit Анна Иванова | Любит кошек и яркие цвета
    """
    
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@dp.message(Command("client"))
async def cmd_client(message: Message):
    """Handle /client command - view client info"""
    if not is_authorized(message):
        return
    
    # Extract client name from command
    text = message.text or ""
    parts = text.split(maxsplit=1)
    
    if len(parts) < 2:
        await message.answer(
            "❌ Укажите имя клиента\n\n"
            "<b>Использование:</b> /client Анна Иванова",
            parse_mode=ParseMode.HTML
        )
        return
    
    client_name = parts[1].strip()
    
    try:
        # Get client info from sheets
        client_info = await sheets_service.get_client(client_name)
        
        if not client_info:
            await message.answer(f"❌ Клиент '{client_name}' не найден")
            return
        
        # Format response
        response = f"📋 <b>Информация о клиенте</b>\n\n"
        response += f"👤 <b>Имя:</b> {client_info['name']}\n"
        
        if client_info.get('instagram'):
            response += f"📱 <b>Instagram:</b> @{client_info['instagram']}\n"
        if client_info.get('telegram'):
            response += f"💬 <b>Telegram:</b> @{client_info['telegram']}\n"
        
        if client_info.get('description'):
            response += f"\n📝 <b>Описание:</b>\n{client_info['description']}\n"
        
        if client_info.get('transactions'):
            response += f"\n💰 <b>История покупок:</b>\n{client_info['transactions']}\n"
        
        if client_info.get('reminder_date'):
            response += f"\n🔔 <b>Напоминание на {client_info['reminder_date']}:</b>\n"
            response += f"{client_info.get('reminder_text', '')}\n"
        
        await message.answer(response, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Error getting client info: {e}")
        await message.answer(f"❌ Ошибка получения данных: {str(e)}")


@dp.message(Command("edit"))
async def cmd_edit_client(message: Message):
    """Handle /edit command - add notes to client"""
    if not is_authorized(message):
        return
    
    # Extract client name and note from command
    text = message.text or ""
    parts = text.split(maxsplit=1)
    
    if len(parts) < 2:
        await message.answer(
            "❌ Неверный формат команды\n\n"
            "<b>Использование:</b> /edit Имя Клиента | Заметка\n\n"
            "<b>Пример:</b> /edit Анна Иванова | Любит кошек и яркие цвета",
            parse_mode=ParseMode.HTML
        )
        return
    
    content = parts[1].strip()
    
    if '|' not in content:
        await message.answer(
            "❌ Используйте символ | для разделения имени и заметки\n\n"
            "<b>Пример:</b> /edit Анна Иванова | Любит кошек",
            parse_mode=ParseMode.HTML
        )
        return
    
    client_name, note = content.split('|', 1)
    client_name = client_name.strip()
    note = note.strip()
    
    if not client_name or not note:
        await message.answer("❌ Укажите имя клиента и заметку")
        return
    
    try:
        # Update client with new note
        client_data = {
            'name': client_name,
            'description': note  # Changed from 'notes' to 'description'
        }
        
        await sheets_service.upsert_client(client_data)
        
        await message.answer(
            f"✅ <b>Заметка добавлена</b>\n\n"
            f"👤 Клиент: {client_name}\n"
            f"📝 Заметка: {note}",
            parse_mode=ParseMode.HTML
        )
        
    except Exception as e:
        logger.error(f"Error editing client: {e}")
        await message.answer(f"❌ Ошибка обновления данных: {str(e)}")


@dp.message(F.voice)
async def handle_voice(message: Message):
    """Handle voice messages"""
    if not is_authorized(message):
        logger.warning(f"Unauthorized voice message from user {message.from_user.id}")
        return
    
    # Send processing message
    processing_msg = await message.answer("🎧 Обрабатываю голосовое сообщение...")
    
    try:
        # Download voice file
        voice_file = await bot.get_file(message.voice.file_id)
        voice_path = f"/tmp/voice_{message.message_id}.ogg"
        await bot.download_file(voice_file.file_path, voice_path)
        
        # Transcribe audio
        transcription = await ai_service.transcribe_audio(voice_path)
        
        # Clean up audio file
        if os.path.exists(voice_path):
            os.remove(voice_path)
        
        if not transcription:
            await processing_msg.edit_text("🤷‍♂️ Не удалось распознать аудио. Попробуйте еще раз.")
            return
        
        logger.info(f"Transcription: {transcription}")
        
        # Process the transcribed text
        await process_message_text(message, processing_msg, transcription)
            
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        await processing_msg.edit_text(f"❌ Ошибка обработки сообщения: {str(e)}")


@dp.message(F.text)
async def handle_text(message: Message):
    """Handle text messages"""
    if not is_authorized(message):
        return
    
    # Skip if it's a command (already handled by other handlers)
    if message.text and message.text.startswith('/'):
        return
    
    # Send processing message
    processing_msg = await message.answer("📝 Обрабатываю сообщение...")
    
    try:
        text = message.text
        logger.info(f"Text message: {text}")
        
        # Process the text
        await process_message_text(message, processing_msg, text)
            
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        await processing_msg.edit_text(f"❌ Ошибка обработки сообщения: {str(e)}")


async def process_message_text(message: Message, processing_msg: Message, text: str):
    """Process message text (from voice or text input)"""
    # Classify message type
    message_type = await ai_service.classify_message(text)
    
    if message_type == "supply":
        await handle_supply(message, processing_msg, text)
    elif message_type == "sale":
        await handle_sale(message, processing_msg, text)
    elif message_type == "preorder":
        await handle_preorder(message, processing_msg, text)
    elif message_type == "client_edit":
        await handle_client_edit(message, processing_msg, text)
    elif message_type == "query":
        await handle_query(message, processing_msg, text)
    else:
        await handle_supply(message, processing_msg, text)


async def handle_supply(message: Message, processing_msg: Message, transcription: str):
    """Handle supply/restock flow"""
    try:
        # Get existing products for fuzzy matching
        existing_products = await sheets_service.get_all_products()
        
        # Parse supply data
        supply_data = await ai_service.parse_supply(transcription, existing_products)
        
        if not supply_data or not supply_data.items:
            await processing_msg.edit_text(
                "❌ Не удалось извлечь информацию о товаре. Попробуйте еще раз с деталями:\n"
                "<i>Название товара, размер и количество</i>",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Update inventory
        items_to_update = [
            {
                'name': item.name,
                'size': item.size,
                'quantity': item.quantity,
                'price': item.price,  # Sale price
                'purchase_price': item.purchase_price  # Purchase price (закупочная цена)
            }
            for item in supply_data.items
        ]
        
        updated_items = await sheets_service.update_inventory(items_to_update, "Supply")
        
        # Format response
        response = "✅ <b>Поставка добавлена</b>\n\n"
        for item in updated_items:
            response += f"📦 {item['name']} (Размер: {item['size']})\n"
            response += f"   Добавлено: {[i.quantity for i in supply_data.items if i.name == item['name'] and i.size == item['size']][0]} шт\n"
            response += f"   Текущий остаток: {item['qty']} шт\n\n"
        
        # Add undo button
        import hashlib
        timestamp = datetime.now().isoformat()
        tx_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🔙 Отменить поставку",
                callback_data=f"undo_supply_{tx_hash}"
            )]
        ])
        
        await processing_msg.edit_text(response, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Error handling supply: {e}")
        await processing_msg.edit_text(f"❌ Ошибка обработки поставки: {str(e)}")


async def handle_sale(message: Message, processing_msg: Message, transcription: str):
    """Handle sale/customer flow - supports multiple items"""
    try:
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch existing products for semantic matching
        existing_products = await sheets_service.get_all_products()
        
        # Parse sale data with product context
        sale_data = await ai_service.parse_sale(transcription, current_date, existing_products)
        
        if not sale_data or not sale_data.items:
            await processing_msg.edit_text(
                "❌ Не удалось извлечь информацию о продаже. Укажите:\n"
                "<i>Имя клиента, товар, размер, количество и цену</i>",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Update inventory for ALL items
        items_to_update = []
        for item in sale_data.items:
            items_to_update.append({
                'name': item.item_name,
                'size': item.size,
                'quantity': item.quantity,
                'price': item.price,
                'client_name': sale_data.client.name
            })
        
        # Try to update inventory - will raise ValueError if item not in stock
        try:
            updated_items = await sheets_service.update_inventory(items_to_update, "Sale")
        except ValueError as e:
            # Stock validation error - show user-friendly message
            await processing_msg.edit_text(
                f"❌ <b>Ошибка продажи</b>\n\n"
                f"{str(e)}",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Update client data
        reminder_date = None
        if sale_data.reminder:
            reminder_date = (datetime.now() + timedelta(days=sale_data.reminder.days_from_now)).strftime('%Y-%m-%d')
        
        # Prepare transaction note for ALL items
        transaction_notes = []
        total_amount = 0
        for item in sale_data.items:
            item_total = item.price * item.quantity
            total_amount += item_total
            transaction_notes.append(
                f"Покупка: {item.item_name} (Размер: {item.size}) x{item.quantity} за ${item_total}"
            )
        
        transaction_note = "; ".join(transaction_notes) + f" от {current_date}"
        
        client_data = {
            'name': sale_data.client.name,
            'instagram': sale_data.client.instagram or '',
            'telegram': sale_data.client.telegram or '',
            'description': sale_data.client.notes or '',  # Client characteristics go to Description
            'transaction': transaction_note,  # Purchase history goes to Transactions
            'reminder_date': reminder_date or '',
            'reminder_text': sale_data.reminder.text if sale_data.reminder else ''
        }
        
        await sheets_service.upsert_client(client_data)
        
        # Format response
        response = "✅ <b>Продажа оформлена</b>\n\n"
        response += f"👤 <b>Клиент:</b> {sale_data.client.name}\n"
        
        if sale_data.client.instagram:
            response += f"📱 Instagram: @{sale_data.client.instagram}\n"
        if sale_data.client.telegram:
            response += f"💬 Telegram: @{sale_data.client.telegram}\n"
        
        response += f"\n💰 <b>Продажа ({len(sale_data.items)} товар(ов)):</b>\n"
        
        # Show each item
        for idx, item in enumerate(sale_data.items, 1):
            item_total = item.price * item.quantity
            response += f"\n{idx}. {item.item_name} (Размер: {item.size})\n"
            response += f"   Количество: {item.quantity} шт\n"
            response += f"   Цена: ${item.price} за шт\n"
            response += f"   Итого: ${item_total}\n"
            
            # Get stock levels for this product
            stock_levels = await sheets_service.get_stock_by_name(item.item_name)
            response += f"   📊 Остаток:\n"
            for stock in stock_levels:
                warning = " ⚠️" if stock['qty'] <= 0 else ""
                response += f"      Размер {stock['size']}: {stock['qty']} шт{warning}\n"
        
        response += f"\n💵 <b>Общая сумма: ${total_amount}</b>\n"
        
        if sale_data.reminder:
            response += f"\n🔔 <b>Напоминание установлено на {reminder_date}:</b>\n"
            response += f"   {sale_data.reminder.text}\n"
        
        # Add undo button (using timestamp as unique identifier)
        timestamp = datetime.now().isoformat()
        # Store transaction data for undo (using timestamp hash for shorter callback_data)
        import hashlib
        tx_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🔙 Отменить продажу",
                callback_data=f"undo_sale_{tx_hash}"
            )]
        ])
        
        await processing_msg.edit_text(response, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Error handling sale: {e}")
        await processing_msg.edit_text(f"❌ Ошибка обработки продажи: {str(e)}")


async def handle_client_edit(message: Message, processing_msg: Message, transcription: str):
    """Handle client information edit flow"""
    try:
        # Parse client edit data
        client_edit_data = await ai_service.parse_client_edit(transcription)
        
        if not client_edit_data:
            await processing_msg.edit_text(
                "❌ Не удалось извлечь информацию о клиенте. Укажите:\n"
                "<i>Имя клиента и заметку/описание</i>",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Update client with new description (not transaction)
        client_data = {
            'name': client_edit_data.client_name,
            'description': client_edit_data.notes  # Goes to Description column
        }
        
        await sheets_service.upsert_client(client_data)
        
        # Format response
        response = "✅ <b>Информация о клиенте обновлена</b>\n\n"
        response += f"👤 <b>Клиент:</b> {client_edit_data.client_name}\n"
        response += f"📝 <b>Добавлено в описание:</b> {client_edit_data.notes}\n"
        
        # Add undo button
        import hashlib
        timestamp = datetime.now().isoformat()
        tx_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🔙 Отменить изменение",
                callback_data=f"undo_client_{client_edit_data.client_name}_{tx_hash}"
            )]
        ])
        
        await processing_msg.edit_text(response, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Error handling client edit: {e}")
        await processing_msg.edit_text(f"❌ Ошибка обновления информации о клиенте: {str(e)}")


async def handle_preorder(message: Message, processing_msg: Message, transcription: str):
    """Обработка предзаказа"""
    try:
        # Parse preorder data
        preorder_data = await ai_service.parse_preorder(transcription)
        
        if not preorder_data or not preorder_data.items:
            await processing_msg.edit_text(
                "❌ Не удалось извлечь информацию о предзаказе. Укажите:\n"
                "<i>Имя клиента, товар, количество, описание</i>",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Prepare items for sheets
        items_to_add = [
            {
                'item_name': item.item_name,
                'quantity': item.quantity,
                'description': item.description
            }
            for item in preorder_data.items
        ]
        
        # Add to Preorders sheet
        success = await sheets_service.add_preorder(
            client_name=preorder_data.client_name,
            items=items_to_add,
            notes=preorder_data.notes
        )
        
        if not success:
            await processing_msg.edit_text(
                "❌ Ошибка добавления предзаказа. Попробуйте еще раз."
            )
            return
        
        # Format response
        response = "✅ <b>Предзаказ добавлен</b>\n\n"
        response += f"👤 <b>Клиент:</b> {preorder_data.client_name}\n\n"
        
        response += f"📝 <b>Товары ({len(preorder_data.items)} поз.):</b>\n"
        for idx, item in enumerate(preorder_data.items, 1):
            response += f"\n{idx}. {item.item_name}\n"
            response += f"   Количество: {item.quantity} шт\n"
            if item.description:
                response += f"   Описание: {item.description}\n"
        
        if preorder_data.notes:
            response += f"\n📌 <b>Заметки:</b> {preorder_data.notes}\n"
        
        response += "\n📋 Предзаказ добавлен в таблицу 'Preorders'"
        
        await processing_msg.edit_text(response, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Error handling preorder: {e}")
        await processing_msg.edit_text(f"❌ Ошибка обработки предзаказа: {str(e)}")


async def handle_query(message: Message, processing_msg: Message, transcription: str):
    """Handle inventory query - show current stock"""
    try:
        # Get all inventory records
        inventory_ws = await sheets_service.spreadsheet.worksheet(sheets_service.INVENTORY_SHEET)
        records = await inventory_ws.get_all_records()
        
        if not records:
            await processing_msg.edit_text(
                "📦 <b>Склад пуст</b>\n\n"
                "Нет товаров на складе.",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Group by product name
        products = {}
        for record in records:
            name = record.get('Name', '')
            size = record.get('Size', '')
            qty = int(record.get('Qty', 0))
            price = float(record.get('Price', 0))
            
            if name:
                if name not in products:
                    products[name] = []
                products[name].append({
                    'size': size,
                    'qty': qty,
                    'price': price
                })
        
        # Format response
        response = "📦 <b>Текущий остаток на складе</b>\n\n"
        
        for product_name, sizes in sorted(products.items()):
            response += f"<b>{product_name}</b>\n"
            for size_info in sorted(sizes, key=lambda x: x['size']):
                qty = size_info['qty']
                size = size_info['size']
                price = size_info['price']
                
                warning = " ⚠️" if qty <= 0 else ""
                price_str = f" (${price})" if price > 0 else ""
                response += f"   • Размер {size}: {qty} шт{price_str}{warning}\n"
            response += "\n"
        
        # Count total items
        total_qty = sum(size_info['qty'] for sizes in products.values() for size_info in sizes)
        response += f"<b>Всего товаров:</b> {total_qty} шт\n"
        response += f"<b>Всего наименований:</b> {len(products)}\n"
        
        await processing_msg.edit_text(response, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        await processing_msg.edit_text(f"❌ Ошибка получения данных о складе: {str(e)}")


@dp.callback_query(F.data.startswith("undo_sale_"))
async def handle_undo_sale(callback: CallbackQuery):
    """Handle undo sale button"""
    if not callback.from_user or not Config.is_user_allowed(callback.from_user.id):
        await callback.answer("Unauthorized", show_alert=True)
        return
    
    try:
        success = await sheets_service.undo_last_sale()
        
        if success:
            await callback.answer("✅ Продажа отменена успешно", show_alert=True)
            
            if callback.message:
                await callback.message.edit_text(
                    f"🔙 <b>Продажа отменена</b>\n\n"
                    f"Последняя транзакция продажи отменена.\n"
                    f"Остатки обновлены.",
                    parse_mode=ParseMode.HTML
                )
        else:
            await callback.answer("❌ Не удалось отменить продажу", show_alert=True)
            
    except Exception as e:
        logger.error(f"Error undoing sale: {e}")
        await callback.answer(f"Ошибка: {str(e)}", show_alert=True)


@dp.callback_query(F.data.startswith("undo_supply_"))
async def handle_undo_supply(callback: CallbackQuery):
    """Handle undo supply button"""
    if not callback.from_user or not Config.is_user_allowed(callback.from_user.id):
        await callback.answer("Unauthorized", show_alert=True)
        return
    
    try:
        success = await sheets_service.undo_last_supply()
        
        if success:
            await callback.answer("✅ Поставка отменена успешно", show_alert=True)
            
            if callback.message:
                await callback.message.edit_text(
                    f"🔙 <b>Поставка отменена</b>\n\n"
                    f"Последняя транзакция поставки отменена.\n"
                    f"Остатки обновлены.",
                    parse_mode=ParseMode.HTML
                )
        else:
            await callback.answer("❌ Не удалось отменить поставку", show_alert=True)
            
    except Exception as e:
        logger.error(f"Error undoing supply: {e}")
        await callback.answer(f"Ошибка: {str(e)}", show_alert=True)


@dp.callback_query(F.data.startswith("undo_client_"))
async def handle_undo_client(callback: CallbackQuery):
    """Handle undo client edit button"""
    if not callback.from_user or not Config.is_user_allowed(callback.from_user.id):
        await callback.answer("Unauthorized", show_alert=True)
        return
    
    try:
        # Extract client name from callback data
        # Format: undo_client_{name}_{hash}
        parts = callback.data.split('_', 3)
        if len(parts) >= 3:
            client_name = parts[2]
        else:
            await callback.answer("❌ Неверный формат данных", show_alert=True)
            return
        
        success = await sheets_service.undo_last_client_update(client_name)
        
        if success:
            await callback.answer("✅ Изменение отменено успешно", show_alert=True)
            
            if callback.message:
                await callback.message.edit_text(
                    f"🔙 <b>Изменение отменено</b>\n\n"
                    f"Последнее изменение для клиента {client_name} отменено.",
                    parse_mode=ParseMode.HTML
                )
        else:
            await callback.answer("❌ Не удалось отменить изменение", show_alert=True)
            
    except Exception as e:
        logger.error(f"Error undoing client edit: {e}")
        await callback.answer(f"Ошибка: {str(e)}", show_alert=True)


@dp.callback_query(F.data.startswith("undo_"))
async def handle_undo_legacy(callback: CallbackQuery):
    """Handle legacy undo button (for backward compatibility)"""
    if not callback.from_user or not Config.is_user_allowed(callback.from_user.id):
        await callback.answer("Unauthorized", show_alert=True)
        return
    
    try:
        # This is for old sale undo buttons without _sale_ prefix
        success = await sheets_service.undo_last_sale()
        
        if success:
            await callback.answer("✅ Продажа отменена успешно", show_alert=True)
            
            if callback.message:
                await callback.message.edit_text(
                    f"🔙 <b>Продажа отменена</b>\n\n"
                    f"Последняя транзакция продажи отменена.\n"
                    f"Остатки обновлены.",
                    parse_mode=ParseMode.HTML
                )
        else:
            await callback.answer("❌ Не удалось отменить продажу", show_alert=True)
            
    except Exception as e:
        logger.error(f"Error undoing sale: {e}")
        await callback.answer(f"Ошибка: {str(e)}", show_alert=True)


async def check_reminders():
    """Check and send due reminders (runs hourly)"""
    try:
        logger.info("Checking for due reminders...")
        reminders = await sheets_service.get_reminders_for_today()
        
        for reminder in reminders:
            try:
                message_text = f"🔔 <b>Напоминание для {reminder['name']}</b>\n\n{reminder['text']}"
                # Send to all allowed users
                for user_id in Config.get_allowed_user_ids():
                    await bot.send_message(
                        chat_id=user_id,
                        text=message_text,
                        parse_mode=ParseMode.HTML
                    )
                
                # Clear the reminder
                await sheets_service.clear_reminder(reminder['name'])
                logger.info(f"Sent and cleared reminder for {reminder['name']}")
                
            except Exception as e:
                logger.error(f"Failed to send reminder for {reminder['name']}: {e}")
        
        if reminders:
            logger.info(f"Processed {len(reminders)} reminders")
        
    except Exception as e:
        logger.error(f"Error checking reminders: {e}")


async def on_startup():
    """Initialize services on startup"""
    logger.info("Starting VoiceStock Bot...")
    
    # Initialize Google Sheets
    await sheets_service.initialize()
    logger.info("Google Sheets service initialized")
    
    # Start scheduler
    scheduler.add_job(
        check_reminders,
        trigger=IntervalTrigger(hours=1),
        id='reminder_check',
        name='Check reminders every hour',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Reminder scheduler started")
    
    logger.info("Bot is ready!")


async def on_shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down bot...")
    scheduler.shutdown()
    await bot.session.close()


async def main():
    """Main entry point"""
    try:
        await on_startup()
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await on_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
