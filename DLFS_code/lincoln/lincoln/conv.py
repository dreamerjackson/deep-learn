import numpy as np
from numpy import ndarray

from .base import ParamOperation


# 输入维度为：(60, 1, 28, 28)
# 经过padding后，维度变为：(60, 1, 38, 38)
# 提取出的小块或子块的维度为：(1156, 60, 1, 5, 5)。
# 输出的维度为：(60, 16, 34, 34)。
class Conv2D_Op(ParamOperation):


    # def _get_image_patches_backward(self, output_grad: ndarray) -> ndarray:
    # # Pad the output gradient
    # padded_output_grad = self._pad_2d_channel_backward(output_grad)
    
    # patches = []
    # img_height = padded_output_grad.shape[2]
    # img_width = padded_output_grad.shape[3]

    # for h in range(img_height-self.param_size+1):
    #     for w in range(img_width-self.param_size+1):
    #         patch = padded_output_grad[:, :, h:h+self.param_size, w:w+self.param_size]
    #         patches.append(patch)
            
    # return np.stack(patches)
    
    # def _pad_2d_channel_backward(self, output_grad: ndarray) -> ndarray:
    #     # This padding is different from the forward pass. 
    #     # We use param_size - 1 padding for the backward pass.
    #     padding_size = self.param_size - 1
    #     pad_width = ((0, 0), (0, 0), (padding_size, padding_size), (padding_size, padding_size))
        
    #     return np.pad(output_grad, pad_width, mode='constant')




    
    # def _get_output_patches(self, output_grad: ndarray, patch_size: int):
    #     '''
    #     Extracts image patches from the output_grad for the input gradient calculation.
    #     '''
    
    #     print("Original shape for patches:", output_grad.shape)
        
    #     # Add padding
    #     padding = 2
    #     output_grad_padded = np.pad(output_grad, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    #     patches = []
    #     img_height = output_grad_padded.shape[2]
    #     img_width = output_grad_padded.shape[3]
    #     for h in range(img_height - patch_size + 1):
    #         for w in range(img_width - patch_size + 1):
    #             patch = output_grad_padded[:, :, h:h+patch_size, w:w+patch_size]
    #             patches.append(patch)
    
    #     expected_patches = (img_height - patch_size + 1) * (img_width - patch_size + 1)
    #     print(f"Expected number of patches: {expected_patches}, Actual number of patches: {len(patches)}")
    #     return np.stack(patches)


    def _get_output_patches(self, output_grad: ndarray, patch_size: int):
        '''
        Extracts image patches from the output_grad for the input gradient calculation.
        '''
    
        # print("Original shape for patches:", output_grad.shape)
    
        patches = []
        img_height = output_grad.shape[2]  # 这应该是34
        img_width = output_grad.shape[3]   # 这也应该是34
    
        max_h_start = output_grad.shape[2] - patch_size  # 这将是 34 - 5 = 29
        max_w_start = output_grad.shape[3] - patch_size  # 同上
        
        for h in range(max_h_start-1):
            for w in range(max_w_start-1):
                patch = output_grad[:, :, h:h+patch_size, w:w+patch_size]
                patches.append(patch)
    
        # expected_patches = 28 * 28
        # print(f"Expected number of patches: {expected_patches}, Actual number of patches: {len(patches)}")
        
        return np.stack(patches)








    
    
    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]  # 获取卷积核的尺寸
        self.param_pad = self.param_size # 计算填充大小

    # 为一维数组添加零填充
    def _pad_1d(self, inp: ndarray) -> ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    # 为批量的一维数组添加零填充
    def _pad_1d_batch(self,
                      inp: ndarray) -> ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)
        
    # 为一个二维观察数据添加零填充
    def _pad_2d_obs(self,
                    inp: ndarray):
        '''
        Input is a 2 dimensional, square, 2D Tensor
        '''
        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate([other, inp_pad, other])


    # def _pad_2d(self,
    #             inp: ndarray):
    #     '''
    #     Input is a 3 dimensional tensor, first dimension batch size
    #     '''
    #     outs = [self._pad_2d_obs(obs, self.param_pad) for obs in inp]
    #
    #     return np.stack(outs)

    # 为通道添加零填充
    def _pad_2d_channel(self,
                        inp: ndarray):
        '''
        inp has dimension [num_channels, image_width, image_height]
        '''
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    # 从输入中提取图像的小块，用于卷积
    def _get_image_patches(self,
                           input_: ndarray):

        # print("Original input shape:", input_.shape)
        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
       # print("Padded input shape:", imgs_batch_pad.shape)
        patches = []
        img_height = imgs_batch_pad.shape[2]
        img_width = imgs_batch_pad.shape[3]
        for h in range(img_height-self.param_size+1):
            for w in range(img_width-self.param_size+1):
                patch = imgs_batch_pad[:, :, h:h+self.param_size, w:w+self.param_size]
                patches.append(patch)
                
        # expected_patches = (img_height - self.param_size + 1) * (img_width - self.param_size + 1)
        # print(f"Expected number of patches: {expected_patches}, Actual number of patches: {len(patches)}")
        return np.stack(patches)

     # 卷积的输出计算函数
    def _output(self,
                inference: bool = False):
        '''
        conv_in: [batch_size, channels, img_width, img_height]
        param: [in_channels, out_channels, fil_width, fil_height]
        '''
    #     assert_dim(obs, 4)
    #     assert_dim(param, 4)
        
        # 获取输入数据的批次大小
        batch_size = self.input_.shape[0]
        # 获取输入图像的高度
        img_height = self.input_.shape[2]
        # print(self.input_.shape[2],self.input_.shape[3])
        # 计算输入图像的大小（宽度 x 高度）
        img_size = self.input_.shape[2] * self.input_.shape[3]
        # 计算参数（卷积核）的大小
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]
        # 从输入数据中提取卷积的子块
        patches = self._get_image_patches(self.input_)
        print(patches.shape)

#         img_width = self.input_.shape[3]
#         img_height = self.input_.shape[2]
# output_width = img_width - self.param_size + 1
# output_height = img_height - self.param_size + 1
# num_patches_width = img_width - self.param_size + 1
# num_patches_height = img_height - self.param_size + 1
# num_patches = num_patches_width * num_patches_height

        out_height = self.input_.shape[2] + 2 * self.param_pad - self.param_size + 1
        out_width =  self.input_.shape[3] + 2 * self.param_pad - self.param_size + 1
       #  print(out_height,out_width)
        # 重新整形子块以匹配卷积核的形状 (60, 34*34, 25)
        patches_reshaped = (patches
                            .transpose(1, 0, 2, 3, 4)
                            .reshape(batch_size, out_height * out_width, -1))
        # 重新整形参数（卷积核）以进行矩阵乘法 (25, 16)
        param_reshaped = (self.param
                          .transpose(0, 2, 3, 1)
                          .reshape(patch_size, -1))
        # 执行矩阵乘法得到卷积操作的输出
        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, out_height, out_width, -1)
            .transpose(0, 3, 1, 2))
        
        # print("output_result:",output_reshaped.shape)
        return output_reshaped


     # 计算输入的梯度
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        img_width = self.input_.shape[3]
        img_height = self.input_.shape[2]
        
        # print("output_grad:",output_grad.shape)
        # print("self.param:",self.param.shape)

        out_height = self.input_.shape[2] + 2 * self.param_pad - self.param_size + 1
        out_width = self.input_.shape[3] + 2 * self.param_pad - self.param_size + 1
        # print(out_height,out_width)

        
        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        mm = self._get_output_patches(output_grad, self.param.shape[2])
        # print("mm:",mm.shape)

        # transposed_mm = mm.transpose(1, 0, 2, 3, 4)
        # print("Transposed mm shape:", transposed_mm.shape)
        # output_patches = (mm.transpose(1, 0, 2, 3, 4)
        #                   .reshape(batch_size * out_height * out_width, -1))
        # param_reshaped = (self.param
        #                   .reshape(self.param.shape[0], -1)
        #                   .transpose(1, 0))

            # 转置 mm 以使 batch_size 成为第一个维度
        transposed_mm = mm.transpose(1, 0, 2, 3, 4)
        print("transposed_mm shape:", transposed_mm.shape) # (60, 784, 16, 5, 5)

        # 计算目标形状并重塑
        num_filters = transposed_mm.shape[2]
        patch_height = transposed_mm.shape[3]
        patch_width = transposed_mm.shape[4]
        # desired_shape = (batch_size * out_height * out_width, num_filters * patch_height * patch_width)
        desired_shape = (transposed_mm.shape[0] * transposed_mm.shape[1], transposed_mm.shape[2] * transposed_mm.shape[3] * transposed_mm.shape[4])
       # print("transposed_mm shape:", transposed_mm.shape)
       # print("desired_shape:", desired_shape.shape)
        output_patches = transposed_mm.reshape(desired_shape) # 60* 784, 16*5*5
       #   print("output_patches shape:", output_patches.shape) 
        param_reshaped = (self.param.reshape(self.param.shape[0], -1).transpose(1, 0))
        print("param_reshaped shape:", param_reshaped.shape) # (400, 1)

        matmul_result = np.matmul(output_patches, param_reshaped)
       # print("matmul_result shape:", matmul_result.shape)
        
        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_width, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )

    # 计算参数（卷积核）的梯度
    # def _param_grad(self, output_grad: ndarray) -> ndarray:
    #     print("_param_grad output_grad",output_grad.shape)

        
    #     out_height = self.input_.shape[2] + 2 * self.param_pad - self.param_size + 1
    #     out_width = self.input_.shape[3] + 2 * self.param_pad - self.param_size + 1
    #     print(out_height,out_width)

    #     batch_size = self.input_.shape[0]
    #     img_size = self.input_.shape[2] * self.input_.shape[3]
    #     in_channels = self.param.shape[0]
    #     out_channels = self.param.shape[1]
    #     in_patches = self._get_output_patches(output_grad, self.param.shape[2])
    #     print(f"zjx-Shape of in_patches: {in_patches.shape}")
    #     print(f"zjx-Expected reshape shape: {batch_size * img_size}")

    #     in_patches_reshape = (
    #         self._get_output_patches(output_grad, self.param.shape[2])
    #         #.reshape(batch_size * img_size, -1)
    #         .reshape(-1, 16*5*5)
    #         .transpose(1,0)
    #         )
    #     print(f"zjx-in_patches_reshape of in_patches: {in_patches_reshape.shape}")

    #     in_channels, out_channels, _, _ = self.param.shape
    #     # Clip output_grad to the center 28x28 part
    #     output_grad = output_grad[:, :, 3:-3, 3:-3]
    
    #     out_grad_reshape = (output_grad
    #                         .transpose(0, 2, 3, 1)
    #                         .reshape(-1, out_channels))
    #     print(f"zjx-out_grad_reshape of in_patches: {out_grad_reshape.shape}")

    #     # return (np.matmul(in_patches_reshape,
    #     #                   out_grad_reshape)
    #     #         .reshape(in_channels, self.param_size, self.param_size, out_channels)
    #     #         .transpose(0, 3, 1, 2))

    #          # Perform matrix multiplication
    #     grad = np.matmul(in_patches_reshape, out_grad_reshape)
        
    #     # Reshape and sum across the batch dimension
    #     grad = grad.reshape(batch_size, in_channels, self.param_size, self.param_size, out_channels)
    #     grad = np.sum(grad, axis=0).transpose(0, 3, 1, 2)
    #     return grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        batch_size = self.input_.shape[0]
        in_channels, out_channels, _, _ = self.param.shape
    
        # Get patches from output_grad
        in_patches = self._get_output_patches(output_grad, self.param.shape[2])
        
        # Reshape in_patches
        in_patches_reshape = in_patches.transpose(1, 0, 2, 3, 4).reshape(batch_size * 28 * 28, -1).transpose(1,0)
        
        # Clip output_grad to the center 28x28 part
        output_grad = output_grad[:, :, 3:-3, 3:-3]
        
        # Reshape output_grad
        out_grad_reshape = output_grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)
    
        # Perform matrix multiplication
        grads_matmul = np.matmul(in_patches_reshape, out_grad_reshape)

        grads_summed = np.sum(grads_matmul, axis=1) 
        # print("grads_summed",grads_summed.shape)
        # Reshape to (in_channels, kernel_height, kernel_width, out_channels)
        grad = grads_summed.reshape(in_channels, self.param_size, self.param_size, out_channels)
    
        return grad.transpose(0, 3, 1, 2)

