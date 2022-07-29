from typing import List

def make_shorter_fn(txt):
    length = len(txt)
    threshold = 500
    if length<threshold:
        return txt
    return txt[0:threshold]
    



def get_sample_txt(make_shorter:bool = False) -> List[str]:
    files = ["my_text.txt", "biden_covid.txt"]
    files = [f"data/{f}" for f in files]
    txts = []
    for myfile in files:
        with open(myfile, 'r', encoding='utf8') as f:
            lines = f.readlines()
        txts.append("".join(lines))

    if make_shorter:
        txts = [make_shorter_fn(t) for t in txts]
    return txts

if __name__ == "__main__":
    txt_list  = get_sample_txt()
    for txt in txt_list:
        print("---"*20)
        print(txt)
