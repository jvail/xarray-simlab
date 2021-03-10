
import sparse
import numpy as np
import xarray as xr

def _is_sparse(arr):
    return isinstance(arr, (sparse.COO, sparse.DOK))

def _sparse_attrs(coo_or_dok, dims):
    return {
        '__xsimlab_sparse_dims__': dims,
        '__xsimlab_sparse_shape__': coo_or_dok.shape,
        '__xsimlab_sparse_fill_value__': coo_or_dok.fill_value.item(),
        '__xsimlab_sparse_format__': 'coo' if isinstance(coo_or_dok, sparse.COO) else 'dok',
        '__xsimlab_sparse__': True
    }

def _sparse_to_zarr_digestible(coo_or_dok, clock=None):
    data = coo_or_dok
    if _is_sparse(coo_or_dok):
        dok = coo_or_dok.asformat('dok')
        types = [(f'dim_{d}', np.int64) for d in range(len(dok.shape))] + [('val', dok.dtype)]
        if clock is not None:
            types = [('clock', np.int64)] + types
            data = np.array([(clock, *idx, val) for idx, val in dok.data.items()], dtype=np.dtype(types))
        else:
            data = np.array([(*idx, val) for idx, val in dok.data.items()], dtype=np.dtype(types))
    return data

def _cover_sparse(ds):
    for name, da in ds.items():
        if _is_sparse(da.data):
            ds = ds.drop(name).assign({
                name: xr.DataArray(
                    data=_sparse_to_zarr_digestible(da.data),
                    attrs={
                        **da.attrs,
                        **_sparse_attrs(da.data, da.dims)
                    }
                )
            })
    return ds

def _recover_sparse(ds):
    for name, da in ds.items():
        if '__xsimlab_sparse__' in da.attrs:
            has_clock_dim = len(da.dims) == 2
            shape = (len(da.data), *da.attrs['__xsimlab_sparse_shape__']) if has_clock_dim else da.attrs['__xsimlab_sparse_shape__']
            dims = (da.dims[0], *da.attrs['__xsimlab_sparse_dims__']) if has_clock_dim else da.attrs['__xsimlab_sparse_dims__']
            fill_value = da.attrs['__xsimlab_sparse_fill_value__']
            _len = len(dims)
            if has_clock_dim:
                data = {tuple(list(v)[:_len]):v[-1] for clock in da.data for v in clock}
            else:
                data = {tuple(list(v)[:_len]):v[-1] for v in da.data}
            data = sparse.DOK(shape, data=data).asformat(da.attrs['__xsimlab_sparse_format__'])
            attrs = {
                name: value for name, value in da.attrs.items() if not name.startswith('__xsimlab_sparse')
            }
            ds = ds.drop(name).assign({
                name: xr.DataArray(
                    data=data,
                    attrs=attrs,
                    dims=dims
                )
            })
    return ds
