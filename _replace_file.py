import os

def replace_file(name_input):
    name = name_input.replace(".jpg","").split("_")  
    print(name)
    try:
        # print(os.listdir(f"record"))
        result = [x for x in os.listdir(f"record") if ("_").join(name) in x]
        return f"{('_').join(result)}"
    except:
        return None
    return None

print(replace_file("1101_22_06_2023_23_52_35_6.jpg"))