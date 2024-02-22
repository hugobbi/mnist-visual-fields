s = "dense_concat1"
match s:
    case _ if 'left' in s:
        print("A string contém 'left'")
    case _ if 'right' in s:
        print("A string contém 'right'")
    case _ if 'concat' in s:
        print("A string contém 'concat'")
    case _:
        print("A string não contém nada")

print('right' in s)