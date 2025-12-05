"""
BrainWave Disorder Detection System - Professional Premium UI
Version: 3.0 - طراحی مدرن و حرفه‌ای
Author: Enhanced by Claude - Premium Edition
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from datetime import datetime
import threading

# تنظیمات ظاهری Premium
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PremiumBrainWaveGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("BrainWave Disorder Detection System - Premium")
        self.window.geometry("1600x950")

        # متغیرها
        self.model = None
        self.file_path = None
        self.current_language = "EN"
        self.is_analyzing = False
        self.theme_mode = "dark"

        # کلاس‌های طبقه‌بندی با ایموجی
        self.classes = {
            "Normal Activity": "🟢",
            "Tumor Area": "🔴",
            "Healthy Area": "💚",
            "Eyes Closed": "😴",
            "Seizure Activity": "⚠️"
        }

        # پالت رنگی Premium
        self.colors = {
            "primary": "#00D9FF",
            "success": "#00FF88",
            "warning": "#FFD700",
            "danger": "#FF3366",
            "info": "#8B5CF6",
            "dark": "#1a1a2e",
            "light": "#edf2f7"
        }

        # دیکشنری ترجمه کامل
        self.translations = {
            "EN": {
                "title": "🧠 BrainWave Disorder Detection System",
                "subtitle": "Advanced AI-Powered Neurological Analysis",
                "select_file": "📁 Select EEG File",
                "analyze": "🔬 Analyze Signal",
                "clear": "🗑️ Clear Results",
                "export": "💾 Export Report",
                "theme": "🌓 Toggle Theme",

                "file_info": "📄 File Information",
                "no_file": "No file selected\n\nClick 'Select EEG File' to begin analysis",
                "file_loaded": "✅ File loaded successfully",
                "file_name": "File Name",
                "file_size": "File Size",
                "shape": "Data Shape",
                "columns": "Columns",
                "rows": "Rows",
                "sample_data": "Sample Data (First 5 values)",

                "results": "📊 Analysis Results",
                "no_results": "No analysis performed yet\n\nLoad a file and click 'Analyze Signal'",
                "prediction": "Diagnosis",
                "confidence": "Confidence Level",
                "class_prob": "Class Probability Distribution",
                "recommendation": "Clinical Recommendation",

                "signal_plot": "EEG Signal - Time Domain Analysis",
                "prob_dist": "Probability Distribution by Class",
                "time_points": "Time Points (samples)",
                "amplitude": "Normalized Amplitude (µV)",
                "probability": "Probability (%)",

                "analyzing": "🔄 Analyzing signal... Please wait",
                "analysis_complete": "✅ Analysis completed successfully!",
                "error": "❌ Error",
                "success": "✅ Success",
                "select_csv": "Please select a CSV file containing EEG data",
                "model_loaded": "Model loaded successfully",
                "model_error": "Failed to load model",

                "low_confidence": "⚠️ Low Confidence Warning",
                "low_conf_msg": "Confidence below 70% - Recommendations:",
                "verify_data": "• Verify signal quality and preprocessing",
                "check_artifacts": "• Check for artifacts or noise",
                "expert_validation": "• Consult with neurological specialist",
                "multiple_tests": "• Consider multiple test sessions",

                "high_confidence": "High confidence - Results are reliable",
                "medium_confidence": "Moderate confidence - Additional validation recommended",
                "low_confidence_short": "Low confidence - Expert review required"
            },

            "DE": {
                "title": "🧠 Gehirnwellen-Störungserkennungssystem",
                "subtitle": "Fortschrittliche KI-gestützte neurologische Analyse",
                "select_file": "📁 EEG-Datei wählen",
                "analyze": "🔬 Signal analysieren",
                "clear": "🗑️ Ergebnisse löschen",
                "export": "💾 Bericht exportieren",
                "theme": "🌓 Design wechseln",

                "file_info": "📄 Dateiinformationen",
                "no_file": "Keine Datei ausgewählt\n\nKlicken Sie auf 'EEG-Datei wählen'",
                "file_loaded": "✅ Datei erfolgreich geladen",
                "file_name": "Dateiname",
                "file_size": "Dateigröße",
                "shape": "Datenform",
                "columns": "Spalten",
                "rows": "Zeilen",
                "sample_data": "Beispieldaten (Erste 5 Werte)",

                "results": "📊 Analyseergebnisse",
                "no_results": "Noch keine Analyse durchgeführt\n\nDatei laden und 'Analysieren' klicken",
                "prediction": "Diagnose",
                "confidence": "Genauigkeitsniveau",
                "class_prob": "Klassenwahrscheinlichkeitsverteilung",
                "recommendation": "Klinische Empfehlung",

                "signal_plot": "EEG-Signal - Zeitbereichsanalyse",
                "prob_dist": "Wahrscheinlichkeitsverteilung nach Klasse",
                "time_points": "Zeitpunkte (Proben)",
                "amplitude": "Normalisierte Amplitude (µV)",
                "probability": "Wahrscheinlichkeit (%)",

                "analyzing": "🔄 Signal wird analysiert... Bitte warten",
                "analysis_complete": "✅ Analyse erfolgreich abgeschlossen!",
                "error": "❌ Fehler",
                "success": "✅ Erfolg",
                "select_csv": "Bitte wählen Sie eine CSV-Datei mit EEG-Daten",
                "model_loaded": "Modell erfolgreich geladen",
                "model_error": "Fehler beim Laden des Modells",

                "low_confidence": "⚠️ Warnung: Geringe Zuverlässigkeit",
                "low_conf_msg": "Genauigkeit unter 70% - Empfehlungen:",
                "verify_data": "• Signalqualität und Vorverarbeitung prüfen",
                "check_artifacts": "• Auf Artefakte oder Rauschen prüfen",
                "expert_validation": "• Neurologischen Facharzt konsultieren",
                "multiple_tests": "• Mehrere Testsitzungen erwägen",

                "high_confidence": "Hohe Zuverlässigkeit - Ergebnisse verlässlich",
                "medium_confidence": "Mittlere Zuverlässigkeit - Zusätzliche Validierung empfohlen",
                "low_confidence_short": "Geringe Zuverlässigkeit - Expertenprüfung erforderlich"
            },

            "FA": {
                "title": "🧠 سیستم تشخیص اختلالات امواج مغزی",
                "subtitle": "تحلیل نورولوژیک پیشرفته با هوش مصنوعی",
                "select_file": "📁 انتخاب فایل EEG",
                "analyze": "🔬 تحلیل سیگنال",
                "clear": "🗑️ پاک کردن نتایج",
                "export": "💾 خروجی گزارش",
                "theme": "🌓 تغییر تم",

                "file_info": "📄 اطلاعات فایل",
                "no_file": "فایلی انتخاب نشده است\n\nروی 'انتخاب فایل EEG' کلیک کنید",
                "file_loaded": "✅ فایل با موفقیت بارگذاری شد",
                "file_name": "نام فایل",
                "file_size": "حجم فایل",
                "shape": "ابعاد داده",
                "columns": "تعداد ستون",
                "rows": "تعداد ردیف",
                "sample_data": "نمونه داده (۵ مقدار اول)",

                "results": "📊 نتایج تحلیل",
                "no_results": "هنوز تحلیلی انجام نشده\n\nفایل را بارگذاری و 'تحلیل' را بزنید",
                "prediction": "تشخیص",
                "confidence": "سطح اطمینان",
                "class_prob": "توزیع احتمال کلاس‌ها",
                "recommendation": "توصیه بالینی",

                "signal_plot": "سیگنال EEG - تحلیل حوزه زمان",
                "prob_dist": "توزیع احتمال بر اساس کلاس",
                "time_points": "نقاط زمانی (نمونه)",
                "amplitude": "دامنه نرمال‌شده (میکروولت)",
                "probability": "احتمال (%)",

                "analyzing": "🔄 در حال تحلیل سیگنال... لطفاً صبر کنید",
                "analysis_complete": "✅ تحلیل با موفقیت انجام شد!",
                "error": "❌ خطا",
                "success": "✅ موفق",
                "select_csv": "لطفاً یک فایل CSV حاوی داده‌های EEG انتخاب کنید",
                "model_loaded": "مدل با موفقیت بارگذاری شد",
                "model_error": "خطا در بارگذاری مدل",

                "low_confidence": "⚠️ هشدار: اطمینان پایین",
                "low_conf_msg": "اطمینان زیر ۷۰٪ - توصیه‌ها:",
                "verify_data": "• کیفیت سیگنال و پیش‌پردازش را بررسی کنید",
                "check_artifacts": "• وجود نویز یا آرتیفکت را چک کنید",
                "expert_validation": "• با متخصص نورولوژی مشورت کنید",
                "multiple_tests": "• جلسات تست مکرر را در نظر بگیرید",

                "high_confidence": "اطمینان بالا - نتایج قابل اعتماد",
                "medium_confidence": "اطمینان متوسط - اعتبارسنجی اضافی توصیه می‌شود",
                "low_confidence_short": "اطمینان پایین - بررسی متخصص ضروری است"
            }
        }

        # بارگذاری مدل
        self.load_model()

        # ساخت UI
        self.create_premium_ui()

    def get_text(self, key):
        """دریافت متن ترجمه‌شده"""
        return self.translations[self.current_language].get(key, key)

    def load_model(self):
        """بارگذاری مدل"""
        try:
            model_path = "models/brainwave_cnn.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ {self.get_text('model_loaded')}")
            else:
                print(f"❌ Model not found at: {model_path}")
                self.model = None
        except Exception as e:
            print(f"❌ {self.get_text('model_error')}: {e}")
            self.model = None

    def change_language(self, lang):
        """تغییر زبان"""
        self.current_language = lang
        self.update_all_texts()

    def toggle_theme(self):
        """تغییر تم"""
        if self.theme_mode == "dark":
            ctk.set_appearance_mode("light")
            self.theme_mode = "light"
        else:
            ctk.set_appearance_mode("dark")
            self.theme_mode = "dark"

    def update_all_texts(self):
        """به‌روزرسانی تمام متن‌ها"""
        self.title_label.configure(text=self.get_text("title"))
        self.subtitle_label.configure(text=self.get_text("subtitle"))

        self.select_btn.configure(text=self.get_text("select_file"))
        self.analyze_btn.configure(text=self.get_text("analyze"))
        self.clear_btn.configure(text=self.get_text("clear"))
        self.theme_btn.configure(text=self.get_text("theme"))

        self.file_info_label.configure(text=self.get_text("file_info"))
        self.results_label.configure(text=self.get_text("results"))

        if not self.file_path:
            self.file_info_text.delete("1.0", "end")
            self.file_info_text.insert("1.0", self.get_text("no_file"))

        if not hasattr(self, 'last_prediction'):
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", self.get_text("no_results"))

    def create_premium_ui(self):
        """ساخت UI Premium"""

        # ========== هدر با گرادیانت ==========
        header_frame = ctk.CTkFrame(
            self.window,
            height=140,
            corner_radius=0,
            fg_color=(self.colors["primary"], self.colors["info"])
        )
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        # دکمه‌های زبان در گوشه راست بالا
        lang_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        lang_frame.place(relx=0.98, rely=0.15, anchor="ne")

        for lang, flag in [("EN", "🇬🇧"), ("DE", "🇩🇪"), ("FA", "🇮🇷")]:
            btn = ctk.CTkButton(
                lang_frame,
                text=f"{flag} {lang}",
                width=85,
                height=38,
                corner_radius=12,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color="#3d3d5c",
                hover_color="#4d4d7a",
                command=lambda l=lang: self.change_language(l)
            )
            btn.pack(side="left", padx=4)

        # عنوان و زیرعنوان
        title_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_container.place(relx=0.5, rely=0.5, anchor="center")

        self.title_label = ctk.CTkLabel(
            title_container,
            text=self.get_text("title"),
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="white"
        )
        self.title_label.pack()

        self.subtitle_label = ctk.CTkLabel(
            title_container,
            text=self.get_text("subtitle"),
            font=ctk.CTkFont(size=14),
            text_color="#d9d9d9"
        )
        self.subtitle_label.pack(pady=(5, 0))

        # ========== دکمه‌های اکشن (Premium Style) ==========
        action_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        action_frame.pack(pady=15)

        # Container داخلی برای مرکز کردن
        btn_container = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_container.pack(expand=True)

        # استایل دکمه‌ها
        btn_config = {
            "height": 50,
            "corner_radius": 15,
            "font": ctk.CTkFont(size=14, weight="bold"),
            "border_width": 0
        }

        self.select_btn = ctk.CTkButton(
            btn_container,
            text=self.get_text("select_file"),
            width=200,
            fg_color=self.colors["info"],
            hover_color="#7c3aed",
            command=self.select_file,
            **btn_config
        )
        self.select_btn.pack(side="left", padx=6)

        self.analyze_btn = ctk.CTkButton(
            btn_container,
            text=self.get_text("analyze"),
            width=200,
            fg_color=self.colors["success"],
            hover_color="#00cc6a",
            command=self.start_analysis_thread,
            **btn_config
        )
        self.analyze_btn.pack(side="left", padx=6)

        self.clear_btn = ctk.CTkButton(
            btn_container,
            text=self.get_text("clear"),
            width=180,
            fg_color=self.colors["danger"],
            hover_color="#cc2952",
            command=self.clear_all,
            **btn_config
        )
        self.clear_btn.pack(side="left", padx=6)

        self.theme_btn = ctk.CTkButton(
            btn_container,
            text=self.get_text("theme"),
            width=150,
            height=50,
            corner_radius=15,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="transparent",
            border_width=2,
            border_color=self.colors["primary"],
            hover_color=self.colors["dark"],
            command=self.toggle_theme
        )
        self.theme_btn.pack(side="left", padx=6)

        # ========== Progress Bar ==========
        self.progress_bar = ctk.CTkProgressBar(
            self.window,
            width=500,
            height=22,
            corner_radius=11,
            progress_color=self.colors["primary"]
        )
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.window,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors["primary"]
        )

        # ========== محتوای اصلی ==========
        content_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # ========== پنل چپ: اطلاعات و نتایج ==========
        left_panel = ctk.CTkFrame(content_frame, width=500, corner_radius=20)
        left_panel.pack(side="left", fill="both", padx=(0, 10), pady=0)
        left_panel.pack_propagate(False)

        # بخش اطلاعات فایل
        file_header = ctk.CTkFrame(left_panel, height=50, corner_radius=15, fg_color=(self.colors["info"], self.colors["primary"]))
        file_header.pack(fill="x", padx=15, pady=(15, 10))
        file_header.pack_propagate(False)

        self.file_info_label = ctk.CTkLabel(
            file_header,
            text=self.get_text("file_info"),
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        )
        self.file_info_label.place(relx=0.5, rely=0.5, anchor="center")

        self.file_info_text = ctk.CTkTextbox(
            left_panel,
            height=240,
            font=ctk.CTkFont(size=13),
            corner_radius=15,
            wrap="word",
            border_width=2,
            border_color=self.colors["info"]
        )
        self.file_info_text.pack(padx=15, pady=(0, 15), fill="x")
        self.file_info_text.insert("1.0", self.get_text("no_file"))

        # بخش نتایج
        results_header = ctk.CTkFrame(left_panel, height=50, corner_radius=15, fg_color=(self.colors["success"], self.colors["primary"]))
        results_header.pack(fill="x", padx=15, pady=(0, 10))
        results_header.pack_propagate(False)

        self.results_label = ctk.CTkLabel(
            results_header,
            text=self.get_text("results"),
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        )
        self.results_label.place(relx=0.5, rely=0.5, anchor="center")

        self.results_text = ctk.CTkTextbox(
            left_panel,
            font=ctk.CTkFont(size=13, family="Consolas"),
            corner_radius=15,
            wrap="word",
            border_width=2,
            border_color=self.colors["success"]
        )
        self.results_text.pack(padx=15, pady=(0, 15), fill="both", expand=True)
        self.results_text.insert("1.0", self.get_text("no_results"))

        # ========== پنل راست: نمودارها ==========
        right_panel = ctk.CTkFrame(content_frame, corner_radius=20)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=0)

        self.plot_frame = ctk.CTkFrame(right_panel, corner_radius=15)
        self.plot_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # پیام خالی اولیه
        welcome_label = ctk.CTkLabel(
            self.plot_frame,
            text="📊\n\nVisualizations will appear here\nafter analysis",
            font=ctk.CTkFont(size=18),
            text_color="gray60"
        )
        welcome_label.place(relx=0.5, rely=0.5, anchor="center")

    def select_file(self):
        """انتخاب فایل"""
        self.file_path = filedialog.askopenfilename(
            title=self.get_text("select_csv"),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if self.file_path:
            try:
                df = pd.read_csv(self.file_path)
                file_size = os.path.getsize(self.file_path) / 1024  # KB

                info = f"✅ {self.get_text('file_loaded')}\n\n"
                info += f"{'='*45}\n"
                info += f"📝 {self.get_text('file_name')}:\n   {os.path.basename(self.file_path)}\n\n"
                info += f"💾 {self.get_text('file_size')}: {file_size:.2f} KB\n\n"
                info += f"📐 {self.get_text('shape')}: {df.shape}\n"
                info += f"   • {self.get_text('columns')}: {df.shape[1]}\n"
                info += f"   • {self.get_text('rows')}: {df.shape[0]}\n\n"
                info += f"🔢 {self.get_text('sample_data')}:\n"
                info += f"   {df.iloc[0, :5].values}\n"
                info += f"{'='*45}"

                self.file_info_text.delete("1.0", "end")
                self.file_info_text.insert("1.0", info)

            except Exception as e:
                messagebox.showerror(self.get_text("error"), str(e))

    def start_analysis_thread(self):
        """شروع تحلیل در Thread"""
        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.progress_bar.pack(pady=10)
        self.progress_label.pack()
        self.progress_label.configure(text=self.get_text("analyzing"))
        self.progress_bar.start()

        thread = threading.Thread(target=self.analyze_signal)
        thread.daemon = True
        thread.start()

    def preprocess_data(self, data):
        """پیش‌پردازش"""
        if isinstance(data, pd.DataFrame):
            if 'y' in data.columns:
                data = data.drop('y', axis=1)
            if 'Unnamed' in str(data.columns):
                data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        if isinstance(data, pd.DataFrame):
            data = data.values

        data = (data - np.mean(data)) / (np.std(data) + 1e-8)

        if len(data.shape) == 1:
            data = data.reshape(1, -1, 1)
        elif len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)

        return data

    def analyze_signal(self):
        """تحلیل سیگنال"""
        if not self.file_path:
            self.window.after(0, lambda: messagebox.showwarning(
                self.get_text("error"),
                self.get_text("select_csv")
            ))
            self.stop_analysis()
            return

        if self.model is None:
            self.window.after(0, lambda: messagebox.showerror(
                self.get_text("error"),
                self.get_text("model_error")
            ))
            self.stop_analysis()
            return

        try:
            df = pd.read_csv(self.file_path)
            data = self.preprocess_data(df)

            predictions = self.model.predict(data, verbose=0)

            if len(predictions.shape) > 1 and predictions.shape[0] > 1:
                avg_predictions = np.mean(predictions, axis=0)
            else:
                avg_predictions = predictions[0]

            predicted_class = np.argmax(avg_predictions)
            confidence = avg_predictions[predicted_class] * 100

            self.last_prediction = {
                'class': predicted_class,
                'confidence': confidence,
                'predictions': avg_predictions
            }

            self.window.after(0, lambda: self.display_premium_results(
                df, avg_predictions, predicted_class, confidence
            ))

            self.window.after(0, lambda: messagebox.showinfo(
                self.get_text("success"),
                self.get_text("analysis_complete")
            ))

        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror(
                self.get_text("error"),
                f"Error: {str(e)}"
            ))
        finally:
            self.stop_analysis()

    def stop_analysis(self):
        """توقف تحلیل"""
        self.is_analyzing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()

    def display_premium_results(self, df, predictions, predicted_class, confidence):
        """نمایش نتایج Premium"""

        class_names = list(self.classes.keys())
        class_name = class_names[predicted_class]
        class_emoji = self.classes[class_name]

        # تعیین سطح اطمینان
        if confidence >= 80:
            conf_level = self.get_text("high_confidence")
            conf_color = "🟢"
        elif confidence >= 60:
            conf_level = self.get_text("medium_confidence")
            conf_color = "🟡"
        else:
            conf_level = self.get_text("low_confidence_short")
            conf_color = "🔴"

        # ساخت متن نتایج
        results = f"\n{'='*50}\n"
        results += f"  {class_emoji} {self.get_text('prediction').upper()}\n"
        results += f"{'='*50}\n\n"
        results += f"  {class_name}\n\n"
        results += f"{'='*50}\n"
        results += f"  {conf_color} {self.get_text('confidence')}: {confidence:.2f}%\n"
        results += f"  {conf_level}\n"
        results += f"{'='*50}\n"
        results += f"\n\n📊 {self.get_text('class_prob')}:\n\n"

        for i, (cls_name, prob) in enumerate(zip(class_names, predictions)):
            emoji = self.classes[cls_name]
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            results += f"  {emoji} {cls_name:20s}\n"
            results += f"     {bar} {prob * 100:.2f}%\n\n"

        # هشدار در صورت اطمینان پایین
        if confidence < 70:
            results += f"\n{'=' * 50}\n"
            results += f"  ⚠️  {self.get_text('low_confidence')}\n"
            results += f"{'=' * 50}\n\n"
            results += f"{self.get_text('low_conf_msg')}\n\n"
            results += f"{self.get_text('verify_data')}\n"
            results += f"{self.get_text('check_artifacts')}\n"
            results += f"{self.get_text('expert_validation')}\n"
            results += f"{self.get_text('multiple_tests')}\n"

        results += f"\n{'=' * 50}\n"
        results += f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        results += f"{'=' * 50}\n"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results)

        # رسم نمودارها
        self.plot_premium_charts(df, predictions, class_names, class_name, confidence)

    def plot_premium_charts(self, df, predictions, class_names, predicted_class, confidence):
        """رسم نمودارهای Premium"""

        # پاک کردن نمودارهای قبلی
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # ایجاد Figure
        fig = Figure(figsize=(12, 10), dpi=100)

        # تنظیم رنگ پس‌زمینه
        if self.theme_mode == "dark":
            bg_color = '#2b2b2b'
            text_color = 'white'
            grid_color = '#404040'
        else:
            bg_color = '#f5f5f5'
            text_color = 'black'
            grid_color = '#d0d0d0'

        fig.patch.set_facecolor(bg_color)

        # ========== نمودار 1: سیگنال EEG ==========
        ax1 = fig.add_subplot(211)
        ax1.set_facecolor(bg_color)

        # نمایش 800 نقطه اول
        signal = df.iloc[0, :800].values if len(df) > 0 else []
        time_points = np.arange(len(signal))

        ax1.plot(time_points, signal, color=self.colors["primary"], linewidth=1.2, alpha=0.9)
        ax1.fill_between(time_points, signal, alpha=0.2, color=self.colors["primary"])

        ax1.set_title(
            self.get_text("signal_plot"),
            color=text_color,
            fontsize=15,
            fontweight='bold',
            pad=15
        )
        ax1.set_xlabel(self.get_text("time_points"), color=text_color, fontsize=12)
        ax1.set_ylabel(self.get_text("amplitude"), color=text_color, fontsize=12)
        ax1.tick_params(colors=text_color)
        ax1.grid(True, alpha=0.3, linestyle='--', color=grid_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color(text_color)
        ax1.spines['bottom'].set_color(text_color)

        # ========== نمودار 2: توزیع احتمالات ==========
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor(bg_color)

        # رنگ‌های کلاس‌ها
        class_colors = [
            self.colors["success"],  # Normal
            self.colors["danger"],  # Tumor
            self.colors["info"],  # Healthy
            self.colors["warning"],  # Eyes Closed
            '#ff00ff'  # Seizure
        ]

        # رسم نمودار میله‌ای افقی
        y_pos = np.arange(len(class_names))
        bars = ax2.barh(y_pos, predictions * 100, height=0.6, color=class_colors, alpha=0.85)

        # اضافه کردن ایموجی و درصد
        for i, (bar, pred, cls_name) in enumerate(zip(bars, predictions, class_names)):
            width = bar.get_width()
            emoji = self.classes[cls_name]

            # ایموجی در سمت چپ
            ax2.text(-5, i, emoji, ha='right', va='center', fontsize=16)

            # درصد در انتهای میله
            ax2.text(
                width + 2, i,
                f'{pred * 100:.1f}%',
                ha='left', va='center',
                color=text_color,
                fontweight='bold',
                fontsize=11
            )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(class_names, color=text_color, fontsize=11)
        ax2.set_xlabel(self.get_text("probability"), color=text_color, fontsize=12, fontweight='bold')
        ax2.set_title(
            self.get_text("prob_dist"),
            color=text_color,
            fontsize=15,
            fontweight='bold',
            pad=15
        )
        ax2.set_xlim(0, 110)
        ax2.tick_params(axis='x', colors=text_color)
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--', color=grid_color)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(text_color)
        ax2.spines['bottom'].set_color(text_color)

        # تنظیم فاصله‌گذاری
        fig.tight_layout(pad=3.0)

        # نمایش در GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def clear_all(self):
        """پاک کردن همه چیز"""
        self.file_path = None

        self.file_info_text.delete("1.0", "end")
        self.file_info_text.insert("1.0", self.get_text("no_file"))

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", self.get_text("no_results"))

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        welcome_label = ctk.CTkLabel(
            self.plot_frame,
            text="📊\n\nVisualizations will appear here\nafter analysis",
            font=ctk.CTkFont(size=18),
            text_color="gray60"
        )
        welcome_label.place(relx=0.5, rely=0.5, anchor="center")

        if hasattr(self, 'last_prediction'):
            delattr(self, 'last_prediction')

    def run(self):
        """اجرای برنامه"""
        self.window.mainloop()

    # ========== اجرای برنامه ==========
if __name__ == "__main__":
    print("🚀 Starting Premium BrainWave Detection System...")
    app = PremiumBrainWaveGUI()
    app.run()