import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from src.util.log import log
from src.settings import CurrentGame
from src.Network import Network


if __name__ == '__main__':
    sample_shape = CurrentGame.representation_shape

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    log_data = []

    for device, dtype, fused, compiled, batch_size in product(
        ['cpu', 'gpu'],
        [torch.float32, torch.float16, torch.bfloat16],
        [True, False],
        ['jit', 'compile', 'none'],
        [1, 8, 32, 64, 128, 256],
    ):
        model = Network(num_res_blocks=10, hidden_size=256, device=torch.device(device))
        model.to(device=device, dtype=dtype)
        model.eval()

        if fused:
            model.fuse_model()

        if compiled == 'jit':
            model = torch.jit.script(model)
        elif compiled == 'compile':
            model = torch.compile(model)

        warmups = 5
        warmup_inputs = [torch.randn((batch_size, *sample_shape), device=device, dtype=dtype) for _ in range(warmups)]
        for warmup_input in warmup_inputs:
            model(warmup_input)

        num_iterations = 256 * 4
        iterations = num_iterations // batch_size

        inputs = [torch.randn((batch_size, *sample_shape), device=device, dtype=dtype) for _ in range(iterations)]

        start = time.time()
        for input in inputs:
            model(input)
        total_time = time.time() - start
        log_data.append(
            {
                'Device': device,
                'DType': dtype,
                'Fused': fused,
                'Compilation': compiled,
                'Batch_Size': batch_size,
                'Iterations': iterations,
                'Total_Time': total_time,
            }
        )
        log(log_data[-1])

    df = pd.DataFrame(log_data)
    print(df.head())

    cuda_grouped = (
        df[(df['Device'] == 'cuda') & (df['DType'] != torch.float32)]
        .groupby(['Fused', 'Compilation', 'Batch_Size'])
        .agg({'Total_Time': ['mean', 'std']})
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    for (fused, compilation), group in cuda_grouped.groupby(['Fused', 'Compilation']):
        plt.errorbar(
            group['Batch_Size'],
            group[('Total_Time', 'mean')],
            yerr=group[('Total_Time', 'std')],
            marker='o',
            label=f'{fused}, {compilation}',
        )

    plt.title('CUDA Float16 Performance (Batch Sizes)')
    plt.xlabel('Batch Size')
    plt.ylabel('Total Time (s)')
    plt.legend(title='Fused & Compilation')
    plt.grid(True)
    plt.show()

    # Analyzing effect of data type for CUDA
    cuda_filtered = df[(df['Device'] == 'cuda')]
    for batch_size in df['Batch_Size'].unique():
        plt.figure(figsize=(10, 6))
        subset = cuda_filtered[cuda_filtered['Batch_Size'] == batch_size]
        for dtype in subset['DType'].unique():
            dtype_data = subset[subset['DType'] == dtype]
            plt.plot(
                dtype_data['Compilation'] + ' | ' + dtype_data['Fused'],
                dtype_data['Total_Time'],
                marker='o',
                label=f'{dtype} (Batch {batch_size})',
            )

        plt.title(f'Effect of Data Type on CUDA Performance (Batch Size {batch_size})')
        plt.xlabel('Compilation | Fused Mode')
        plt.ylabel('Total Time (s)')
        plt.xticks(rotation=45)
        plt.legend(title='Data Type')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
