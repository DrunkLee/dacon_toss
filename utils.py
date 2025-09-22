import inspect
import os
import psutil

def class_print(obj, context:str):
    '''
    obj: self 같은 인스턴스를 넣어주면 자동으로 class/method명을 가져옵니다.
    context: 출력할 메세지
    '''
    class_name = obj.__class__.__name__
    method_name = inspect.stack()[1].function
    print(f"[{class_name}] [{method_name}] {context}")
    
def print_sys_usage(note:str = ''):
    '''현재 CPU 메모리 사용량 및 여분을 로깅합니다.'''
    process = psutil.Process(os.getpid())
    
    mem_info = process.memory_info()
    rss = mem_info.rss / (1024 ** 2)
    vms = mem_info.vms / (1024 ** 2)
    
    sys_mem = psutil.virtual_memory()
    total = sys_mem.total / (1024 ** 2)
    available = sys_mem.available / (1024 ** 2)
    
    print("=" * 50)
    print(f"[SYSTEM USAGE] {note}")
    print(f"Process Memory: {rss:.2f} MB (RSS) / {vms:.2f} MB (VMS)")
    print(f"System Memory : {available:.2f} MB free / {total:.2f} MB total")
    print("=" * 50)