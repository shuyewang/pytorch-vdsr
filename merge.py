import h5py
import glob
import tables as tb

with tb.File(r'D:\merge_248.h5',mode='w') as h5fw:
    row1 = 0
    for h5name in glob.glob(r'D:\train248\*.h5'):
        print(h5name)
        h5fr = tb.File(h5name,mode='r') 
        dset_data = h5fr.root.data[:]
        dset_label = h5fr.root.label[:]
        if row1 == 0 :
            # earr_data = dset_data
            # earr_label = dset_label
            earr_data = h5fw.create_earray(h5fw.root,'data', 
                                     shape=(0,1,41,41), obj=dset_data )
            earr_label = h5fw.create_earray(h5fw.root,'label', 
                                     shape=(0,1,41,41), obj=dset_label )
        else :
            earr_data.append(dset_data)
            earr_label.append(dset_label)
        row1 += earr_data.shape[0] 