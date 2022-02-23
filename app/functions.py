from csv import writer as csv_writer

from matplotlib import pyplot as plt


def save_load_time(filepath: str, start: float, end: float):
    template = 'Load time (s): {}'
    try:
        with open(filepath, 'r') as f:
            file_data = f.read()
            time_value = float(file_data.split(':')[-1].strip())

        with open(filepath, 'w') as f:
            f.write(template.format(time_value + (end - start)))

    except FileNotFoundError:
        with open(filepath, 'w') as f:
            f.write(template.format(end - start))


def save_bar(filepath: str,
             labels: list,
             a_values: list,
             b_values: list,
             a_label: str = 'A values',
             b_label: str = 'B values',
             title: str = 'Title',
             y_label: str = 'y label',
             vertical_labels: bool = False,
             size_inches: tuple = None,
             bar_width: float = .4):

    x = [x for x in range(len(labels))]
    fig, ax = plt.subplots()
    if size_inches is not None:
        fig.set_size_inches(*size_inches)

    ax.bar([x - bar_width / 2 for x in x], a_values, bar_width, label=a_label)
    ax.bar([x + bar_width / 2 for x in x], b_values, bar_width, label=b_label)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    rotation = 'vertical' if vertical_labels else 'horizontal'
    ax.set_xticklabels(labels, rotation=rotation)
    ax.legend()

    fig.tight_layout()
    plt.savefig(filepath)


def convert_result(result: list, cutout_word: str, a_key, b_key):
    res_dict = dict()
    for el in result:
        region, year, score = el
        region = region.replace(cutout_word, '').strip()
        if region not in res_dict:
            res_dict[region] = dict()
        res_dict[region][year] = score

    labels = list(res_dict.keys())
    a_values, b_values = list(), list()
    for el in labels:
        a_values.append(res_dict[el][a_key])
        b_values.append(res_dict[el][b_key])

    return labels, a_values, b_values


def save_csv(data: list, header: list, filepath: str, encoding: str, delimiter: str):
    with open(filepath, 'w', encoding=encoding) as f:
        file_writer = csv_writer(f, delimiter=delimiter)
        file_writer.writerow(list(map(str, header)))
        for el in data:
            file_writer.writerow(list(map(str, el)))
