class GNUtils:


    def __init__(self):
        pass

    #
    def short_transformed(self, all_ax, transformed):
        print("short_transformed... ")
        new_t = []
        for ax, st in zip(all_ax, transformed):
            if ax == 0:
                new_t.append(st)
            else:
                new_t.append(st[0])
        print("short_transformed... done")
        return new_t

    #
    def serialize(self, data):
        import flax.serialization
        # Wandelt den State in einen Byte-String um
        print("serialize", type(data))
        if isinstance(data, list):
            for i, item in enumerate(data):
                print(f"item {i}", type(item))
                if isinstance(item, list):
                    for j, item2 in enumerate(item):
                        print(f"item {j}", type(item2))
        binary_data = flax.serialization.to_bytes(data)
        return binary_data