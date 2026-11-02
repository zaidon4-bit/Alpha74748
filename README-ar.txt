# αlpha AI — Qwen3.7 Max + Gemma 4

تطبيق دردشة Flask متعدد المستخدمين، مع مفاتيح API منفصلة لكل نموذج ولوحة تحكم مشرف. يدعم التشغيل المحلي على SQLite والنشر على Render مع PostgreSQL في Supabase.

## النماذج الظاهرة للمستخدم

1. **alpha 3.1 max** — Qwen3.7 Max: `qwen/qwen3.7-max:free` عبر Xkiro أو مسار API متوافق.
2. **alpha 2.7 flash** — Gemma 4 26B A4B IT: `gemma-4-26b-a4b-it` عبر Google GenAI أو مسار API متوافق.

## المميزات المحفوظة

- لوحة مشرف ومستخدمون منفصلون، وتخصيص مفاتيح API لكل مستخدم.
- حتى 8 مسارات API لكل نموذج بترتيب أولوية، وتحويل تلقائي للمسار الاحتياطي عند فشل المسار الحالي.
- System Prompt مستقل لكل نموذج، محفوظ في قاعدة البيانات ومتاح للمشرف فقط.
- زر Thinking لـ Qwen لعرض reasoning الذي يعيده مزود API عند تفعيله؛ لا يضمن هذا التحكم في التفكير الداخلي للمزود.
- مستويات التفكير لـ Gemma، وWeb Search، وUNLAZY.
- مراجعة الكود وإصلاحه تلقائياً، Diff Viewer، والمرفقات والصور والملفات.
- ذاكرة محادثة مشتركة بين النموذجين لآخر 80 رسالة لكل مستخدم.

## التشغيل المحلي

```bash
python -m pip install -r requirements.txt
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD='ضع-كلمة-مرور-قوية-هنا'
python app.py
```

افتح `http://127.0.0.1:5000/login`.

## النشر على Render + Supabase

راجع الدليل التفصيلي: `DEPLOY-RENDER-SUPABASE-AR.md`.

تنبيه: Render Free قد يوقف الخدمة بعد 15 دقيقة من عدم النشاط، وSupabase Free قد يوقف قاعدة المشروع عند انخفاض نشاطها على مدى أسبوع. راجع قيود الخطتين قبل الاعتماد على الموقع لمستخدمين دائمين.

تستخدم نسخة Render متغير `DATABASE_URL` للاتصال بـ Supabase PostgreSQL. يتم إنشاء الجداول تلقائياً. لا ترفع `app.db` أو `.api_encryption.key` أو `.flask_secret_key` أو `.env` إلى GitHub.

اسم المستخدم الافتراضي للمشرف: `admin`. كلمة المرور تحددها أنت في متغير Render `ADMIN_PASSWORD` ولا تُحفظ داخل المستودع.

## ترحيل قاعدة SQLite القديمة

توجد أداة اختيارية `migrate_sqlite_to_supabase.py` لنقل الحسابات والمحادثات وتعليمات النماذج ومفاتيح API المشفرة ومسارات API من `app.db` إلى Supabase. راجع قسم الترحيل في دليل النشر، واستخدم قيمة `.api_encryption.key` الأصلية كمتغير `API_ENCRYPTION_KEY` حتى تبقى المفاتيح القديمة قابلة لفك التشفير.

## الفحص الصحي

المسار `/healthz` يفحص اتصال التطبيق بقاعدة البيانات. يجب أن يعيد HTTP 200 مع `{"status":"ok"}`.
