import jax
import jax.numpy as jnp


HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='config')


def unsqueeze_like(x, *objs):
    # coding=utf-8
    # Copyright 2023 The Google Research Authors.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """Append additional axes to each obj in objs for each extra axis in x.

    Example: x of shape (bs,n,c) and s,t both of shape (bs,),
    sp,tp = unsqueeze_like(x,s,t) has sp and tp of shape (bs,1,1)

    Args:
      x: ndarray with shape that to unsqueeze like
      *objs: ndarrays to unsqueeze to that shape

    Returns:
      unsqueeze_objs: unsqueezed versions of *objs
    """
    if len(objs) != 1:
        return [
            unsqueeze_like(x, obj) for obj in objs
        ]  # pytype: disable=bad-return-type  # jax-ndarray
    elif hasattr(objs[0], "shape") and len(objs[0].shape):  # broadcast to x shape
        return objs[0][(Ellipsis,) + len(x.shape[1:]) * (None,)]
    else:
        return objs[
            0
        ]  # if it is a scalar, it already broadcasts to x shape  # pytype: disable=bad-return-type  # jax-ndarray


@jax.jit
def relative_error(x, y, axis=-1):
    """
    Compute |x - y|/(|x| + |y|) with L1 norm over the axis `axis`.
    """
    return (
        jnp.abs(x - y).sum(axis)
        / (jnp.abs(x).sum(axis) + jnp.abs(y).sum(axis))
    )


