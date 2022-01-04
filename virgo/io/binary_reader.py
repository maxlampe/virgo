import numpy as np

"""
    Reads data from a vine binary file into an array of dictionaries.

    Parameters:
        filebase                    # Name of the snapshot file

    Returns:
         pos                        # position array
         shock_normal               # shock normal array
         mach                       # mach vector
         hsml                       # hsml vector
"""
def read_binary_data(filebase):

    filename = filebase + '_pos.dat'

    Npart = np.fromfile(filename,           # filename to read
                        count=1,            # we only want to read one integer
                        dtype=np.int64      # we want to read a 64 byte integer, aka a C long.
                        )[0]                # only read first entry


    print('Reading ', Npart, ' particles')

    """
        Read positions
    """
    pos = np.reshape(np.fromfile(filename,             # file to read
                          offset=8,           # skip first numer -> number of particles
                          count=Npart*3,      # how many particles to read
                          dtype=np.float32    # read doubles -> 64bit float
                          ), (Npart, 3) )

    """
        Read shock normal
    """
    filename = filebase + '_shock_normal.dat'
    shock_normal = np.reshape(np.fromfile(filename,             # file to read
                          offset=8,           # skip first numer -> number of particles
                          count=Npart*3,      # how many particles to read
                          dtype=np.float32    # read doubles -> 64bit float
                          ), (Npart, 3) )

    """
        Read Mach
    """
    filename = filebase + '_mach.dat'
    mach = np.fromfile( filename,             # file to read
                        offset=8,           # skip first numer -> number of particles
                        count=Npart,      # how many particles to read
                        dtype=np.float32    # read doubles -> 64bit float
                        )

    """
        Read HSML
    """
    filename = filebase + '_hsml.dat'
    hsml = np.fromfile(filename,             # file to read
                          offset=8,           # skip first numer -> number of particles
                          count=Npart,      # how many particles to read
                          dtype=np.float32    # read doubles -> 64bit float
                          )

    return np.vstack((  pos[:,0], pos[:,1], pos[:,2], 
                        shock_normal[:,0], shock_normal[:,1], shock_normal[:,2], 
                        mach, hsml))
