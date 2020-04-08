import matlab.engine


def main():
    eng = matlab.engine.start_matlab()
    filepath1 = input("请输入载体图的路径:\n")
    filepath2 = input("请输入水印图路径:\n")
    filepath3 = "after_insert.png"
    eng.insert(filepath1, filepath2, filepath3, nargout=0) 
    print("执行结束!")
    eng.quit()
if __name__ == "__main__":
    main()    























