import threading
import time


# def thread_job():
#     print("A thread, number is %s" % threading.current_thread())
#
# def test():
#     # print("__name__", __name__)
#     # print(threading.active_count())  # 线程的数目
#     # print(threading.enumerate())  # 活跃线程的枚举对象
#     # print(threading.current_thread())  # 当前线程
#     thread_1 = threading.Thread(target=thread_job)
#     thread_1.start()
#
# if __name__ == "__main__":
#     test()


#  使用join 可以控制多个线程的执行顺序  这点与java很像

# def thread_job():
#     print("T1 start\n")
#     for i in range(10):
#         time.sleep(1)  # 任务间隔1秒
#     print("T1 finish\n")
#
# def T2_job():
#     print("T2 start\n")
#     print("T2 finish\n")
#
#
# thread_1 = threading.Thread(target=thread_job, name="T1")
# thread_2 = threading.Thread(target=T2_job, name="T2")
# thread_1.start()
# thread_2.start()
# thread_2.join()
# thread_1.join()
# print("all done\n")


from queue import Queue


def job(a, q):
    for i in range(len(a)):
        a[i] = a[i]**2
    q.put(a)

def multi_threading():
    q = Queue()
    threads = []
    data = [
        [1,2,3],
        [4,5,6],
        [7,8,9],
    ]
    for i in range(3):
        t = threading.Thread(target=job, args=(data[i], q))
        t.start()
        threads.append(t)

    for tr in threads:
        tr.join()  # 主线程等子线程执行完，在结束

    results = []
    for _ in range(len(data)):
        results.append(q.get())
    print(results)

if __name__ == "__main__":
    multi_threading()
