tags2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
         'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
         'B': 1, 'I': 2, 'E': 3, 'S': 4}
def spans_to_tags(spans, seq_len):
    tags = [tags2id['O']] * seq_len
    for span in spans:
        tags[span[0]] = tags2id['B']
        tags[span[0]:span[1]+1] = [tags2id['I']] * (span[1]-span[0]+1)
        if span[0] == span[1]:
            tags[span[0]] = tags2id['S']
        else:
            tags[span[0]] = tags2id['B']
            tags[span[1]] = tags2id['E']
    return tags


def get_arg_span(bioes_tags):
    start, end = None, None
    arguments = []
    in_entity_flag = False
    for idx, tag in enumerate(bioes_tags):
        if in_entity_flag == False:
            if tag == 1: # B
                in_entity_flag = True
                start = idx
            elif tag == 4: # S
                start = idx
                end = idx
                arguments.append((start, end))
                start = None
                end = None
        else:
            if tag == 0: # O
                in_entity_flag = False
                start = None
                end = None
            elif tag == 1: # B
                in_entity_flag = True
                start = idx
            elif tag == 3: # E
                in_entity_flag = False
                end = idx
                arguments.append((start, end))
                start = None
                end = None
            elif tag == 4: # S
                in_entity_flag = False
                start = idx
                end = idx
                arguments.append((start, end))
                start = None
                end = None
    return arguments

def extract_arguments(bioes_list):
    arguments_list = []
    for pred_tags in bioes_list:
        arguments = get_arg_span(pred_tags)
        arguments_list.append(arguments)
    return arguments_list

# def extract_arguments(bio_list):
#     arguments_list = []
#     for pred_tags in bio_list:
#         start, end = None, None
#         arguments = []
#         in_entity_flag = False
#         for idx, tag in enumerate(pred_tags):
#             if in_entity_flag:
#                 if tag == 1:
#                     end = idx - 1
#                     arguments.append((start, end))
#                     start = idx
#                     end = None
#                 elif tag == 0:
#                     end = idx - 1
#                     arguments.append((start, end))
#                     start = None
#                     end = None
#                     in_entity_flag = False
#             else:
#                 if tag == 1:
#                     in_entity_flag = True
#                     start = idx
#         arguments_list.append(arguments)
#     return arguments_list
