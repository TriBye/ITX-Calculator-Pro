import tkinter as tk
from tkinter import ttk, messagebox

# Conversion factors relative to base units
UNIT_CATEGORIES = {
    "Length": {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mile": 1609.34
    },
    "Mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 0.000001,
        "lb": 0.453592,
        "oz": 0.0283495
    },
    "Time": {
        "s": 1.0,
        "ms": 0.001,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0
    },
    "Area": {
        "m^2": 1.0,
        "cm^2": 0.0001,
        "mm^2": 0.000001,
        "km^2": 1_000_000.0,
        "ft^2": 0.092903,
        "acre": 4046.86
    },
    "Volume": {
        "m^3": 1.0,
        "cm^3": 0.000001,
        "L": 0.001,
        "mL": 0.000001,
        "ft^3": 0.0283168,
        "gal": 0.00378541
    }
}

TEMPERATURE_UNITS = ["C", "F", "K"]

CURRENCY_RATES = {
    "EUR": 1.0,
    "USD": 1.08,
    "GBP": 0.86,
    "JPY": 169.5,
    "CHF": 0.95
}

UNIT_NAMES = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mile": "mile",
    "kg": "kilogram",
    "g": "gram",
    "mg": "milligram",
    "lb": "pound",
    "oz": "ounce",
    "s": "second",
    "ms": "millisecond",
    "min": "minute",
    "hr": "hour",
    "day": "day",
    "m^2": "square meter",
    "cm^2": "square centimeter",
    "mm^2": "square millimeter",
    "km^2": "square kilometer",
    "ft^2": "square foot",
    "acre": "acre",
    "m^3": "cubic meter",
    "cm^3": "cubic centimeter",
    "L": "liter",
    "mL": "milliliter",
    "ft^3": "cubic foot",
    "gal": "gallon",
    "C": "degree Celsius",
    "F": "degree Fahrenheit",
    "K": "Kelvin"
}

COMBINABLE_UNITS = sorted({unit for units in UNIT_CATEGORIES.values() for unit in units} | set(TEMPERATURE_UNITS))


class UnitConverterApp:
    def __init__(self, master):
        self.master = master
        master.title("Unit Converter")
        master.geometry("520x420")
        master.resizable(False, False)

        notebook = ttk.Notebook(master)
        notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.unit_tab = ttk.Frame(notebook, padding=15)
        self.currency_tab = ttk.Frame(notebook, padding=15)
        self.combine_tab = ttk.Frame(notebook, padding=15)

        notebook.add(self.unit_tab, text="Units")
        notebook.add(self.currency_tab, text="Currency")
        notebook.add(self.combine_tab, text="Combine Units")

        self._build_unit_tab()
        self._build_currency_tab()
        self._build_combine_tab()

    # --- Unit conversion tab ---
    def _build_unit_tab(self):
        self.category_var = tk.StringVar(value="Length")
        category_label = ttk.Label(self.unit_tab, text="Category")
        category_label.grid(row=0, column=0, sticky="w")
        self.category_combo = ttk.Combobox(
            self.unit_tab,
            textvariable=self.category_var,
            values=list(UNIT_CATEGORIES.keys()) + ["Temperature"],
            state="readonly"
        )
        self.category_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
        self.category_combo.bind("<<ComboboxSelected>>", self._update_unit_options)

        ttk.Label(self.unit_tab, text="Value").grid(row=1, column=0, sticky="w")
        self.unit_value_var = tk.StringVar()
        self.unit_value_entry = ttk.Entry(self.unit_tab, textvariable=self.unit_value_var)
        self.unit_value_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(self.unit_tab, text="From").grid(row=2, column=0, sticky="w")
        self.from_unit_var = tk.StringVar()
        self.from_unit_combo = ttk.Combobox(self.unit_tab, textvariable=self.from_unit_var, state="readonly")
        self.from_unit_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(self.unit_tab, text="To").grid(row=3, column=0, sticky="w")
        self.to_unit_var = tk.StringVar()
        self.to_unit_combo = ttk.Combobox(self.unit_tab, textvariable=self.to_unit_var, state="readonly")
        self.to_unit_combo.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=5)

        convert_btn = ttk.Button(self.unit_tab, text="Convert", command=self.convert_units)
        convert_btn.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        self.unit_result_var = tk.StringVar(value="Result will appear here")
        ttk.Label(self.unit_tab, textvariable=self.unit_result_var, foreground="#333333").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        self.unit_tab.columnconfigure(1, weight=1)
        self._update_unit_options()

    def _update_unit_options(self, *_args):
        category = self.category_var.get()
        if category == "Temperature":
            options = TEMPERATURE_UNITS
        else:
            options = list(UNIT_CATEGORIES[category].keys())
        self.from_unit_combo.configure(values=options)
        self.to_unit_combo.configure(values=options)
        if options:
            self.from_unit_var.set(options[0])
            self.to_unit_var.set(options[-1])

    def convert_units(self):
        value_text = self.unit_value_var.get().strip()
        if not value_text:
            messagebox.showwarning("Input Required", "Enter a value to convert.")
            return

        try:
            value = float(value_text)
        except ValueError:
            messagebox.showerror("Invalid Value", "Please enter a numeric value.")
            return

        category = self.category_var.get()
        from_unit = self.from_unit_var.get()
        to_unit = self.to_unit_var.get()

        if category == "Temperature":
            result = self._convert_temperature(value, from_unit, to_unit)
        else:
            factors = UNIT_CATEGORIES.get(category)
            if not factors or from_unit not in factors or to_unit not in factors:
                messagebox.showerror("Conversion Error", "Unsupported unit selection for this category.")
                return
            base_value = value * factors[from_unit]
            result = base_value / factors[to_unit]

        self.unit_result_var.set(f"{value} {from_unit} = {result:.6g} {to_unit}")

    def _convert_temperature(self, value, from_unit, to_unit):
        if from_unit == to_unit:
            return value

        if from_unit == "C":
            celsius = value
        elif from_unit == "F":
            celsius = (value - 32.0) * 5.0 / 9.0
        elif from_unit == "K":
            celsius = value - 273.15
        else:
            raise ValueError("Unsupported temperature unit")

        if to_unit == "C":
            return celsius
        if to_unit == "F":
            return celsius * 9.0 / 5.0 + 32.0
        if to_unit == "K":
            return celsius + 273.15
        raise ValueError("Unsupported temperature unit")

    # --- Currency tab ---
    def _build_currency_tab(self):
        ttk.Label(self.currency_tab, text="Amount").grid(row=0, column=0, sticky="w")
        self.currency_value_var = tk.StringVar()
        ttk.Entry(self.currency_tab, textvariable=self.currency_value_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5
        )

        ttk.Label(self.currency_tab, text="From").grid(row=1, column=0, sticky="w")
        self.currency_from_var = tk.StringVar(value="EUR")
        self.currency_from_combo = ttk.Combobox(
            self.currency_tab,
            textvariable=self.currency_from_var,
            values=list(CURRENCY_RATES.keys()),
            state="readonly"
        )
        self.currency_from_combo.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(self.currency_tab, text="To").grid(row=2, column=0, sticky="w")
        self.currency_to_var = tk.StringVar(value="USD")
        self.currency_to_combo = ttk.Combobox(
            self.currency_tab,
            textvariable=self.currency_to_var,
            values=list(CURRENCY_RATES.keys()),
            state="readonly"
        )
        self.currency_to_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Button(self.currency_tab, text="Convert", command=self.convert_currency).grid(
            row=3, column=0, columnspan=2, pady=10, sticky="ew"
        )

        self.currency_result_var = tk.StringVar(value="Converted amount will appear here")
        ttk.Label(self.currency_tab, textvariable=self.currency_result_var, foreground="#333333").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        self.currency_tab.columnconfigure(1, weight=1)

    def convert_currency(self):
        value_text = self.currency_value_var.get().strip()
        if not value_text:
            messagebox.showwarning("Input Required", "Enter an amount to convert.")
            return
        try:
            amount = float(value_text)
        except ValueError:
            messagebox.showerror("Invalid Value", "Please enter a numeric amount.")
            return

        from_code = self.currency_from_var.get()
        to_code = self.currency_to_var.get()
        from_rate = CURRENCY_RATES.get(from_code)
        to_rate = CURRENCY_RATES.get(to_code)
        if from_rate is None or to_rate is None:
            messagebox.showerror("Conversion Error", "Unsupported currency selection.")
            return

        eur_amount = amount / from_rate
        result = eur_amount * to_rate
        self.currency_result_var.set(f"{amount:.2f} {from_code} = {result:.2f} {to_code}")

    # --- Combination tab ---
    def _build_combine_tab(self):
        ttk.Label(self.combine_tab, text="First Unit").grid(row=0, column=0, sticky="w")
        self.combine_first_var = tk.StringVar(value=COMBINABLE_UNITS[0])
        self.combine_first_combo = ttk.Combobox(
            self.combine_tab,
            textvariable=self.combine_first_var,
            values=COMBINABLE_UNITS,
            state="readonly"
        )
        self.combine_first_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(self.combine_tab, text="Operation").grid(row=1, column=0, sticky="w")
        self.operation_var = tk.StringVar(value="*")
        operations_frame = ttk.Frame(self.combine_tab)
        operations_frame.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Radiobutton(operations_frame, text="Multiply", value="*", variable=self.operation_var).pack(side=tk.LEFT)
        ttk.Radiobutton(operations_frame, text="Divide", value="/", variable=self.operation_var).pack(side=tk.LEFT, padx=10)

        ttk.Label(self.combine_tab, text="Second Unit").grid(row=2, column=0, sticky="w")
        self.combine_second_var = tk.StringVar(value=COMBINABLE_UNITS[1])
        self.combine_second_combo = ttk.Combobox(
            self.combine_tab,
            textvariable=self.combine_second_var,
            values=COMBINABLE_UNITS,
            state="readonly"
        )
        self.combine_second_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Button(self.combine_tab, text="Combine", command=self.combine_units).grid(
            row=3, column=0, columnspan=2, pady=10, sticky="ew"
        )

        self.combine_result_var = tk.StringVar(value="Combine two units to see the result")
        ttk.Label(self.combine_tab, textvariable=self.combine_result_var, foreground="#333333").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        self.combine_tab.columnconfigure(1, weight=1)

    def combine_units(self):
        first = self.combine_first_var.get()
        second = self.combine_second_var.get()
        op = self.operation_var.get()
        result_symbol, description = self._combine_symbolic(first, op, second)
        self.combine_result_var.set(f"{first} {op} {second} -> {result_symbol} ({description})")

    def _combine_symbolic(self, first, op, second):
        if op == "*":
            if first == second:
                symbol = f"{first}^2"
            else:
                symbol = f"{first}{second}"
            description = f"{UNIT_NAMES.get(first, first)} multiplied by {UNIT_NAMES.get(second, second)}"
        else:
            if first == second:
                symbol = "dimensionless"
                description = "dimensionless quantity"
            else:
                symbol = f"{first}/{second}"
                description = f"{UNIT_NAMES.get(first, first)} divided by {UNIT_NAMES.get(second, second)}"
        return symbol, description


def main():
    root = tk.Tk()
    app = UnitConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
