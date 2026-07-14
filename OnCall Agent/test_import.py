import sys
try:
    print("尝试导入 app.main...")
    import app.main
    print("✅ app.main 导入成功！")
except Exception as e:
    print(f"❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
