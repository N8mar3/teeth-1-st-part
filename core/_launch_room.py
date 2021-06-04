if __name__ == '__main__':
    import json
    from _main_distribution_room import train_model, work
    train_model(*list(json.load(open('training_3', 'r')).values()))
    # work(*list(json.load(open('working', 'r')).values()))
