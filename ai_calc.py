import tkinter as tk
from tkinter import scrolledtext, messagebox, font as tkfont
import google.generativeai as genai
import os
import threading
from dotenv import load_dotenv
import sympy

# --- Configure Gemini API ---
load_dotenv()

# Note: Model initialization is now handled within the AIAssistantApp class

class AIAssistantApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Math Assistant (Gemini Chat)") # Updated title
        master.geometry("600x500")


        self.model = None # Will be initialized in _initialize_model_and_chat
        self.chat = None  # Will be initialized in _initialize_model_and_chat
        self.sympy_locals = {
            'x': sympy.Symbol('x'),
            'y': sympy.Symbol('y'),
            'pi': sympy.pi,
            'e': sympy.E,
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'sqrt': sympy.sqrt,
            'log': sympy.log,
            'ln': sympy.log,
            'exp': sympy.exp,
            'I': sympy.I
        }

        # --- Styling ---
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.default_font_family = self.default_font.actual()["family"]
        self.text_font = (self.default_font_family, 11)
        self.entry_font = (self.default_font_family, 11)
        self.button_font = (self.default_font_family, 10, "bold")

        # --- Chat Display Area ---
        self.chat_display = scrolledtext.ScrolledText(
            master, 
            wrap=tk.WORD, 
            state='disabled', 
            font=self.text_font,
            padx=5,
            pady=5,
            relief=tk.SUNKEN,
            borderwidth=1
        )

        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            master,
            textvariable=self.status_var,
            anchor='w',
            font=(self.default_font_family, 9),
            fg="#555555"
        )
        self.status_label.pack(fill=tk.X, padx=12, pady=(0, 4))

        # --- Input Frame ---
        input_frame = tk.Frame(master, pady=5)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.input_entry = tk.Entry(input_frame, font=self.entry_font, width=50)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", self.send_message_event)

        self.send_button = tk.Button(
            input_frame, 
            text="Send", 
            command=self.send_message, 
            font=self.button_font,
            bg="#4CAF50", fg="white",
            relief=tk.RAISED,
            padx=10
        )
        self.send_button.pack(side=tk.RIGHT)

        self._initialize_model_and_chat() # Initialize model and chat session

        if not self.model or not self.chat: # Check both model and chat
            self.display_message("Gemini API is unavailable. I will use offline evaluation when possible.", "System")
            self.status_var.set("Offline mode")
        else:
            # Updated initial message for clarity
            self.display_message("Ask me anything! I will provide plain text answers.", "AI")
            self.status_var.set("" if self.model and self.chat else "Offline mode")

    def _initialize_model_and_chat(self):
        try:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=GEMINI_API_KEY)

            system_instruction_text = (
                "You are an AI Assistant. Your responses must be plain text only. "
                "Do not use any markdown formatting (like bold, italics, lists, code blocks, headers, etc.). "
                "Provide only the raw text of your answer."
            )
            
            # Use the model name from your original code
            self.model = genai.GenerativeModel(
                'gemini-2.0-flash', # As per your code
                system_instruction=system_instruction_text
            )
            self.chat = self.model.start_chat(history=[])
            print("Gemini model and chat initialized successfully with 'gemini-2.0-flash'.")

        except Exception as e:
            messagebox.showerror("API Configuration Error", 
                                 f"Error configuring Gemini API: {e}\n"
                                 "Please ensure google-generativeai is installed, "
                                 "GEMINI_API_KEY is set, and the model 'gemini-2.0-flash' is accessible.")
            self.model = None 
            self.chat = None
            print(f"Error during Gemini initialization: {e}")





    def display_message(self, message, sender="AI"):
        self._append_message(sender, message)

    def _append_message(self, sender, message):
        self.chat_display.config(state='normal')
        if self.chat_display.index('end-1c') != '1.0':
            self.chat_display.insert(tk.END, "\n")
        formatted_message = f"{sender}: {message}\n"
        self.chat_display.insert(tk.END, formatted_message)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)


    def _evaluate_locally(self, user_input):
        try:
            expr = sympy.sympify(user_input, locals=self.sympy_locals)
        except Exception:
            return None
        if expr.free_symbols:
            return None
        numeric = sympy.N(expr, 12)
        return f"Result: {numeric}"

    def _generate_response(self, user_input):
        local_result = self._evaluate_locally(user_input)
        if local_result is not None:
            return local_result
        if not self.model or not self.chat:
            raise RuntimeError("Remote AI model is not available.")
        response = self.chat.send_message(user_input)
        return getattr(response, 'text', str(response))

    def send_message_event(self, event):
        self.send_message()

    def send_message(self):
        user_input = self.input_entry.get().strip()
        if not user_input:
            return

        self.input_entry.delete(0, tk.END)
        self.display_message(user_input, "You")
        self.send_button.config(state='disabled')
        self.input_entry.config(state='disabled')
        self.status_var.set("Processing...")

        worker = threading.Thread(
            target=self._get_gemini_response,
            args=(user_input,),
            daemon=True
        )
        worker.start()

    def _get_gemini_response(self, user_input):
        try:
            ai_message = self._generate_response(user_input)
        except RuntimeError as err:
            ai_message = f"Sorry, {err}"
        except Exception as e:
            ai_message = f"Sorry, an error occurred while generating a response: {e}"
            print(f"Gemini processing error: {e}")

        self.master.after(0, self._update_ui_with_response, user_input, ai_message)


    def _update_ui_with_response(self, user_prompt, ai_message):
        if not ai_message:
            ai_message = "No response generated."
        self._append_message("AI", ai_message)
        self.status_var.set("" if self.model and self.chat else "Offline mode")
        self.send_button.config(state='normal')
        self.input_entry.config(state='normal')
        self.input_entry.focus()


if __name__ == "__main__":
    root = tk.Tk()
    app = AIAssistantApp(root)
    root.mainloop()
