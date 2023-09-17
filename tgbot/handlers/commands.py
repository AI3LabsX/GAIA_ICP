"""
The module contains handlers that respond to commands from bot users

Handlers:
    start_cmd_from_admin_handler    - response to the /start command from the bot administrator
    start_cmd_from_user_handler     - response to the /start command from the bot user
    help_cmd_handler                - response to the /help command

Note:
    Handlers are imported into the __init__.py package handlers,
    where a tuple of HANDLERS is assembled for further registration in the application
"""
import os
import shutil
from collections import deque
from pathlib import Path

import openai
from langchain.embeddings import OpenAIEmbeddings
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import ContextTypes, CommandHandler, CallbackContext

from tgbot import PROJECT_ROOT
from tgbot.config import BOT_LOGO
from tgbot.handlers.messages import get_vectorstore_icp, generate_response
from tgbot.utils.environment import env
from tgbot.utils.filters import is_admin_filter
from tgbot.utils.templates import template


async def start_cmd_from_admin(update: Update, context: CallbackContext) -> None:
    """Handles command /start from the admin"""
    welcome_message = "👋 Hello, Admin! I am your ChatBot manager. What would you like to do?"
    # TODO: Fix it Model selection (See ChatGPT)!!!
    buttons = [
        [KeyboardButton(text="⚙ ICP bot")]
    ]
    keyboard = ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text(text=welcome_message, reply_markup=keyboard)

    # Set the conversation state to 'idle'
    context.user_data['conversation_state'] = 'idle'


async def start_cmd_from_user(update: Update, context: CallbackContext) -> None:
    """Handles command /start from the user"""
    username: str = update.message.from_user.first_name

    welcome_message = f"👋 Hello, {username if username else 'user'}! I am your ChatBot manager. What would you like " \
                      f"to do? "

    buttons = [
        [KeyboardButton(text="⚙ ICP bot")]
    ]
    keyboard = ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text(text=welcome_message, reply_markup=keyboard)

    # Set the conversation state to 'idle'
    context.user_data['conversation_state'] = 'idle'


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles command /help from the user

    Note:
        In this handler as an example, we will use the template renderer to format the response message
    """
    data: dict = {
        "framework_url": "https://python-telegram-bot.org",
        "licence_url": "https://github.com/rin-gil/python-telegram-bot-template/blob/master/LICENCE",
        "author_profile_url": "https://www.instagram.com/rexxar.ai/",
        "author_email": "riharex420@gmail.com",
    }
    caption: str = await template.render(template_name="help_cmd.jinja2", data=data)
    await update.message.reply_photo(photo=BOT_LOGO, caption=caption)


async def add_docs(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add documents to the chat."""
    await update.message.reply_text('Send a document to add to the chat.')


async def exit_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Exit the chat and return to the main menu."""
    await update.message.reply_text(text="Exiting the chat. Returning to the main menu.")
    # Set the conversation state to 'idle'
    context.user_data['conversation_state'] = 'idle'

    if os.path.exists(f"files/vectorstore_{update.effective_chat.id}"):
        try:
            shutil.rmtree(f"files/vectorstore_{update.effective_chat.id}")
        except Exception as ex:
            print(ex)

    await start_cmd_from_user(update, context)


openai.api_key = env.get_open_ai_api()

history = deque(maxlen=50)
embeddings = OpenAIEmbeddings()
faiss = Path(PROJECT_ROOT).joinpath("knowladge_base", "faiss_index_ICP")
vectorstore = get_vectorstore_icp(str(faiss), embeddings)


# Main Function for user handle to get ChatBot functionality!
async def get_chat_gpt(update: Update, context) -> None:
    user_query = update.message.text.replace("/icp", "").strip()

    results = generate_response(user_query, history, vectorstore)
    history.append((user_query, results))
    print(results)
    print(history)
    await update.message.reply_text(text=results, parse_mode=None)


# Creating handlers
start_cmd_from_admin_handler: CommandHandler = CommandHandler(
    command="start", callback=start_cmd_from_admin, filters=is_admin_filter
)

start_cmd_from_user_handler: CommandHandler = CommandHandler(command="start", callback=start_cmd_from_user)
help_cmd_handler: CommandHandler = CommandHandler(command="help", callback=help_cmd)
icp_cmd_handler: CommandHandler = CommandHandler(command="icp", callback=get_chat_gpt)
