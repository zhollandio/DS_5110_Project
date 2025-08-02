import fmi


def fmi_communicator(world_size, rank):

    communicator = fmi.Communicator(rank, world_size, "/tmp/Anomaly Detection/config/fmi.json", "fmi_pair", 512)

    if communicator is None:
        print("unable to create FMI Communicator")
        return

    communicator.hint(fmi.hints.fast)

    print("retrieved fmi communicator")

    return communicator
