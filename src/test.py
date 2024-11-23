from collections import Counter 
data = [
    {
        "bro" : 1,
        "man" : 2, 
        "yo"  : 5, 
        "aooo": 1, 
    }, 
    {
        "girl" : 1,
        "qo" : 2, 
        "asd"  : 5, 
        "vxc": 1, 
    }, 
]
data_list = [] 
for d in data: 
    data_list += list(d.values()) 
counter = Counter(data_list) 
print(counter.items()) 