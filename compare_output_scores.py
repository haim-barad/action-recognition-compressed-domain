import sys

def file2dict(file):
    d = {}

    with open(file) as f:
        for line in f:
           #print("line is:::",line)
           key = line.split(':')[1].split()[0]
           #print(key)
           val = line.split(':')[2].split()[0]
           d[key] = val
        
    return d
    
def compare(standard,compress_add,compress = None,residuals=None):

    standard_d = file2dict(standard)
    compress_add_d = file2dict(compress_add)
    if compress:
        compress_d = file2dict(compress)
    if residuals:
        residuals_d = file2dict(residuals)

    with open("comparison.txt",'w+') as f:
        for key,val in standard_d.items():
           
            f.write(key)
            f.write(":")
            f.write(str(val))
            f.write(",")
            f.write(str(compress_add_d[key]))
            if compress:
                f.write(",")
                f.write(str(compress_d[key]))
            if residuals:
                f.write(",")
                f.write(str(residuals_d[key]))
            f.write("\n")
    
        f.close()
        
if __name__ == "__main__":
    standard     = sys.argv[1] 
    compress_add = sys.argv[2]
    compress     = sys.argv[2]
    residuals    = sys.argv[3]
    #if "residual" in compress:
        #residuals = compress
    compare(standard,compress_add,compress,residuals)
    
    
       