function [dim] = hdf5getdim(fn, varName)
    info = h5info(fn);
    dim = [];
    for i = 1:size(info.Datasets,1)
        dat = info.Datasets(i);
        if(strcmp(dat.Name, varName))
            dim = dat.Dataspace.Size;
        end
    end
    assert(all(size(dim)>0));
end
