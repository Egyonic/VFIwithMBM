import cupy
import torch
import re
import math



kernel_Kernel_updateOutput = '''
	extern "C" __global__ void kernel_Kernel_updateOutput(
		const int n,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* weight,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblOutput = 0.0;

		const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX      = ( intIndex                                                    ) % SIZE_3(output);
		

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
			    float delta_x = OFFSET_4(offset_y, intSample, intFilterY*SIZE_1(vertical) + intFilterX, intY, intX);
			    float delta_y = OFFSET_4(offset_x, intSample, intFilterY*SIZE_1(horizontal) + intFilterX, intY, intX);
			    
			    float position_x = delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1;
			    float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
			    if (position_x < 0)
			        position_x = 0;
			    if (position_x > SIZE_3(input) - 1)
			        position_x = SIZE_3(input) - 1;
			    if (position_y < 0)
			        position_y = 0;
			    if (position_y > SIZE_2(input) - 1)
			        position_y =  SIZE_2(input) - 1;
			    
			    int left = floor(delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1);
			    int right = left + 1;
			    if (left < 0)
			        left = 0;
			    if (left > SIZE_3(input) - 1)
			        left = SIZE_3(input) - 1;
			    if (right < 0)
			        right = 0;
			    if (right > SIZE_3(input) - 1)
			        right = SIZE_3(input) - 1;
			    
			    int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
			    int bottom = top + 1;
			    if (top < 0)
			        top = 0;
			    if (top > SIZE_2(input) - 1)
			        top =  SIZE_2(input) - 1;
			    if (bottom < 0)
			        bottom = 0;   
			    if (bottom > SIZE_2(input) - 1)
			        bottom = SIZE_2(input) - 1;
			    
			    float floatValue = VALUE_4(input, intSample, intDepth, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			                       VALUE_4(input, intSample, intDepth, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			                       VALUE_4(input, intSample, intDepth, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			                       VALUE_4(input, intSample, intDepth, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			                       
				dblOutput += floatValue * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(weight, intSample, SIZE_1(vertical)*intFilterY + intFilterX, intY, intX);
			}
		}
		output[intIndex] = dblOutput;
	} }
'''

kernel_Kernel_updateGradVertical = '''
	extern "C" __global__ void kernel_Kernel_updateGradVertical(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* weight,
		float* gradVertical
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) ) % SIZE_0(gradVertical);
		const int intFilterY  = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                        ) % SIZE_1(gradVertical);
		const int intY        = ( intIndex / SIZE_3(gradVertical)                                               ) % SIZE_2(gradVertical);
		const int intX        = ( intIndex                                                                      ) % SIZE_3(gradVertical);

		for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1){
		    int intDepth = intFilterY * SIZE_1(horizontal) + intFilterX;
		    float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
			float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);
			
			float position_x = delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1;
			float position_y = delta_y + intY + intFilterY - (SIZE_1(horizontal) - 1) / 2 + 1;
			if (position_x < 0)
			    position_x = 0;
			if (position_x > SIZE_3(input) - 1)
			    position_x = SIZE_3(input) - 1;
			if (position_y < 0)
			    position_y = 0;
			if (position_y > SIZE_2(input) - 1)
			    position_y =  SIZE_2(input) - 1;
		
			int left = floor(delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1);
			int right = left + 1;
			if (left < 0)
			    left = 0;
			if (left > SIZE_3(input) - 1)
			    left = SIZE_3(input) - 1;
			if (right < 0)
			    right = 0;
			if (right > SIZE_3(input) - 1)
			    right = SIZE_3(input) - 1;

			int top = floor(delta_y + intY + intFilterY - (SIZE_1(horizontal) - 1) / 2 + 1);
			int bottom = top + 1;
			if (top < 0)
			    top = 0;
			if (top > SIZE_2(input) - 1)
			    top =  SIZE_2(input) - 1;
			if (bottom < 0)
			    bottom = 0;   
			if (bottom > SIZE_2(input) - 1)
			    bottom = SIZE_2(input) - 1;
			
			float floatSampled0 = VALUE_4(input, intSample, 0, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled1 = VALUE_4(input, intSample, 1, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled2 = VALUE_4(input, intSample, 2, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			
			floatOutput += VALUE_4(gradLoss, intSample, 0, intY, intX) * floatSampled0 * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(weight, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 1, intY, intX) * floatSampled1 * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(weight, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 2, intY, intX) * floatSampled2 * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(weight, intSample, intDepth, intY, intX);
		}
		gradVertical[intIndex] = floatOutput;
	} }

'''

kernel_Kernel_updateGradHorizontal = '''
	extern "C" __global__ void kernel_Kernel_updateGradHorizontal(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* offset_x,
		const float* offset_y,
		const float* weight,
		float* gradHorizontal
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
		const int intFilterX  = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                          ) % SIZE_1(gradHorizontal);
		const int intY        = ( intIndex / SIZE_3(gradHorizontal)                                                   ) % SIZE_2(gradHorizontal);
		const int intX        = ( intIndex                                                                            ) % SIZE_3(gradHorizontal);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1){
		    int intDepth = intFilterY * SIZE_1(vertical) + intFilterX;
		    float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
			float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);
		
			float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
			float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
			if (position_x < 0)
			    position_x = 0;
			if (position_x > SIZE_3(input) - 1)
			    position_x = SIZE_3(input) - 1;
			if (position_y < 0)
			    position_y = 0;
			if (position_y > SIZE_2(input) - 1)
			    position_y =  SIZE_2(input) - 1;
		
			int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
			int right = left + 1;
			if (left < 0)
			    left = 0;
			if (left > SIZE_3(input) - 1)
			    left = SIZE_3(input) - 1;
			if (right < 0)
			    right = 0;
			if (right > SIZE_3(input) - 1)
			    right = SIZE_3(input) - 1;

			int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
			int bottom = top + 1;
			if (top < 0)
			    top = 0;
			if (top > SIZE_2(input) - 1)
			    top =  SIZE_2(input) - 1;
			if (bottom < 0)
			    bottom = 0;   
			if (bottom > SIZE_2(input) - 1)
			    bottom = SIZE_2(input) - 1;
			
			float floatSampled0 = VALUE_4(input, intSample, 0, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled1 = VALUE_4(input, intSample, 1, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled2 = VALUE_4(input, intSample, 2, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
				
			floatOutput += VALUE_4(gradLoss, intSample, 0, intY, intX) * floatSampled0 * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(weight, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 1, intY, intX) * floatSampled1 * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(weight, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 2, intY, intX) * floatSampled2 * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(weight, intSample, intDepth, intY, intX);
		}
		gradHorizontal[intIndex] = floatOutput;
	} }
'''

kernel_Kernel_updateGradWeight = '''
	extern "C" __global__ void kernel_Kernel_updateGradWeight(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		float* gradWeight
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	    float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight) / SIZE_1(gradWeight) ) % SIZE_0(gradWeight);
		const int intDepth    = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight)                    ) % SIZE_1(gradWeight);
		const int intY        = ( intIndex / SIZE_3(gradWeight)                                       ) % SIZE_2(gradWeight);
		const int intX        = ( intIndex                                                          ) % SIZE_3(gradWeight);
		
		int intFilterY = intDepth / SIZE_1(vertical);
        int intFilterX = intDepth % SIZE_1(vertical);
        
        float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
		float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);
		
		float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
		float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
		if (position_x < 0)
			position_x = 0;
		if (position_x > SIZE_3(input) - 1)
			position_x = SIZE_3(input) - 1;
		if (position_y < 0)
			position_y = 0;
		if (position_y > SIZE_2(input) - 1)
			position_y =  SIZE_2(input) - 1;
		
		int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
		int right = left + 1;
		if (left < 0)
			left = 0;
		if (left > SIZE_3(input) - 1)
			left = SIZE_3(input) - 1;
		if (right < 0)
			right = 0;
		if (right > SIZE_3(input) - 1)
			right = SIZE_3(input) - 1;

		int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
		int bottom = top + 1;
		if (top < 0)
			top = 0;
		if (top > SIZE_2(input) - 1)
			top =  SIZE_2(input) - 1;
		if (bottom < 0)
			bottom = 0;   
		if (bottom > SIZE_2(input) - 1)
			bottom = SIZE_2(input) - 1;
		
		for (int intChannel = 0; intChannel < 3; intChannel++){
		    floatOutput += VALUE_4(gradLoss, intSample, intChannel, intY, intX) * (
		                   VALUE_4(input, intSample, intChannel, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, intChannel, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, intChannel, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, intChannel, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y))
		                   ) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
		} 
		gradWeight[intIndex] = floatOutput;
	} }
'''

kernel_Kernel_updateGradOffsetX = '''
	extern "C" __global__ void kernel_Kernel_updateGradOffsetX(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* weight,
		float* gradOffsetX
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	    float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX) / SIZE_1(gradOffsetX) ) % SIZE_0(gradOffsetX);
		const int intDepth    = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX)                       ) % SIZE_1(gradOffsetX);
		const int intY        = ( intIndex / SIZE_3(gradOffsetX)                                             ) % SIZE_2(gradOffsetX);
		const int intX        = ( intIndex                                                                   ) % SIZE_3(gradOffsetX);

		int intFilterY = intDepth / SIZE_1(vertical);
        int intFilterX = intDepth % SIZE_1(vertical);

        float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
		float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);

		float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
		float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
		if (position_x < 0)
			position_x = 0;
		if (position_x > SIZE_3(input) - 1)
			position_x = SIZE_3(input) - 1;
		if (position_y < 0)
			position_y = 0;
		if (position_y > SIZE_2(input) - 1)
			position_y =  SIZE_2(input) - 1;
		
		int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
		int right = left + 1;
		if (left < 0)
			left = 0;
		if (left > SIZE_3(input) - 1)
			left = SIZE_3(input) - 1;
		if (right < 0)
			right = 0;
		if (right > SIZE_3(input) - 1)
			right = SIZE_3(input) - 1;

		int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
		int bottom = top + 1;
		if (top < 0)
			top = 0;
		if (top > SIZE_2(input) - 1)
			top =  SIZE_2(input) - 1;
		if (bottom < 0)
			bottom = 0;   
		if (bottom > SIZE_2(input) - 1)
			bottom = SIZE_2(input) - 1;

		for (int intChannel = 0; intChannel < 3; intChannel++){
			floatOutput += VALUE_4(gradLoss, intSample, intChannel, intY, intX) * (
		                   - VALUE_4(input, intSample, intChannel, top, left)  * (1 + (left - position_x))
		                   - VALUE_4(input, intSample, intChannel, top, right)  *  (1 - (right - position_x))
			               + VALUE_4(input, intSample, intChannel, bottom, left) * (1 + (left - position_x))
			               + VALUE_4(input, intSample, intChannel, bottom, right) * (1 - (right - position_x))
			               )
		                   * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX)
		                   * VALUE_4(weight, intSample, intDepth, intY, intX);
		} 
		gradOffsetX[intIndex] = floatOutput;
	} }
'''

kernel_Kernel_updateGradOffsetY = '''
	extern "C" __global__ void kernel_Kernel_updateGradOffsetY(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* weight,
		float* gradOffsetY
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	    float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX) / SIZE_1(gradOffsetX) ) % SIZE_0(gradOffsetX);
		const int intDepth    = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX)                       ) % SIZE_1(gradOffsetX);
		const int intY        = ( intIndex / SIZE_3(gradOffsetX)                                             ) % SIZE_2(gradOffsetX);
		const int intX        = ( intIndex                                                                   ) % SIZE_3(gradOffsetX);

		int intFilterY = intDepth / SIZE_1(vertical);
        int intFilterX = intDepth % SIZE_1(vertical);

        float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
		float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);

		float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
		float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
		if (position_x < 0)
			position_x = 0;
		if (position_x > SIZE_3(input) - 1)
			position_x = SIZE_3(input) - 1;
		if (position_y < 0)
			position_y = 0;
		if (position_y > SIZE_2(input) - 1)
			position_y =  SIZE_2(input) - 1;
		
		int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
		int right = left + 1;
		if (left < 0)
			left = 0;
		if (left > SIZE_3(input) - 1)
			left = SIZE_3(input) - 1;
		if (right < 0)
			right = 0;
		if (right > SIZE_3(input) - 1)
			right = SIZE_3(input) - 1;

		int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
		int bottom = top + 1;
		if (top < 0)
			top = 0;
		if (top > SIZE_2(input) - 1)
			top =  SIZE_2(input) - 1;
		if (bottom < 0)
			bottom = 0;   
		if (bottom > SIZE_2(input) - 1)
			bottom = SIZE_2(input) - 1;

		for (int intChannel = 0; intChannel < 3; intChannel++){
		    floatOutput += VALUE_4(gradLoss, intSample, intChannel, intY, intX) * (
		                   - VALUE_4(input, intSample, intChannel, top, left)  * (1 + (top - position_y)) 
		                   + VALUE_4(input, intSample, intChannel, top, right)  *  (1 + (top - position_y)) 
			               - VALUE_4(input, intSample, intChannel, bottom, left) * (1 - (bottom - position_y)) 
			               + VALUE_4(input, intSample, intChannel, bottom, right) * (1 - (bottom - position_y))
			               )
		                   * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX)
		                   * VALUE_4(weight, intSample, intDepth, intY, intX);
		} 
		gradOffsetY[intIndex] = floatOutput;
	} }
'''


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    while True:
        objectMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel


# end

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end

class FunctionKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vertical, horizontal, offset_x, offset_y, weight):
        ctx.save_for_backward(input, vertical, horizontal, offset_x, offset_y, weight)

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert (intInputHeight == intOutputHeight + intFilterSize - 1)
        assert (intInputWidth == intOutputWidth + intFilterSize - 1)

        assert (input.is_contiguous() == True)
        assert (vertical.is_contiguous() == True)
        assert (horizontal.is_contiguous() == True)
        assert (offset_x.is_contiguous() == True)
        assert (offset_y.is_contiguous() == True)
        assert (weight.is_contiguous() == True)

        output = input.new_zeros([intSample, intInputDepth, intOutputHeight, intOutputWidth])

        if input.is_cuda == True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('kernel_Kernel_updateOutput', cupy_kernel('kernel_Kernel_updateOutput', {
                'input': input,
                'vertical': vertical,
                'horizontal': horizontal,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'weight': weight,
                'output': output
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), offset_x.data_ptr(), offset_y.data_ptr(),
                      weight.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end

    @staticmethod
    def backward(ctx, gradOutput):
        input, vertical, horizontal, offset_x, offset_y, weight = ctx.saved_tensors

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert (intInputHeight == intOutputHeight + intFilterSize - 1)
        assert (intInputWidth == intOutputWidth + intFilterSize - 1)

        assert (gradOutput.is_contiguous() == True)

        gradInput = input.new_zeros([intSample, intInputDepth, intInputHeight, intInputWidth]) if \
            ctx.needs_input_grad[0] == True else None
        gradVertical = input.new_zeros([intSample, intFilterSize, intOutputHeight, intOutputWidth]) if \
            ctx.needs_input_grad[1] == True else None
        gradHorizontal = input.new_zeros([intSample, intFilterSize, intOutputHeight, intOutputWidth]) if \
            ctx.needs_input_grad[2] == True else None
        gradOffsetX = input.new_zeros([intSample, intFilterSize * intFilterSize, intOutputHeight, intOutputWidth]) if \
            ctx.needs_input_grad[3] == True else None
        gradOffsetY = input.new_zeros([intSample, intFilterSize * intFilterSize, intOutputHeight, intOutputWidth]) if \
            ctx.needs_input_grad[4] == True else None
        gradWeight = input.new_zeros([intSample, intFilterSize * intFilterSize, intOutputHeight, intOutputWidth]) if \
            ctx.needs_input_grad[5] == True else None

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            nv = gradVertical.nelement()
            cupy_launch('kernel_Kernel_updateGradVertical', cupy_kernel('kernel_Kernel_updateGradVertical', {
                'gradLoss': gradOutput,
                'input': input,
                'horizontal': horizontal,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'weight': weight,
                'gradVertical': gradVertical
            }))(
                grid=tuple([int((nv + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[nv, gradOutput.data_ptr(), input.data_ptr(), horizontal.data_ptr(), offset_x.data_ptr(),
                      offset_y.data_ptr(), weight.data_ptr(), gradVertical.data_ptr()],
                stream=Stream
            )

            nh = gradHorizontal.nelement()
            cupy_launch('kernel_Kernel_updateGradHorizontal', cupy_kernel('kernel_Kernel_updateGradHorizontal', {
                'gradLoss': gradOutput,
                'input': input,
                'vertical': vertical,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'weight': weight,
                'gradHorizontal': gradHorizontal
            }))(
                grid=tuple([int((nh + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[nh, gradOutput.data_ptr(), input.data_ptr(), vertical.data_ptr(), offset_x.data_ptr(),
                      offset_y.data_ptr(), weight.data_ptr(), gradHorizontal.data_ptr()],
                stream=Stream
            )

            nx = gradOffsetX.nelement()
            cupy_launch('kernel_Kernel_updateGradOffsetX', cupy_kernel('kernel_Kernel_updateGradOffsetX', {
                'gradLoss': gradOutput,
                'input': input,
                'vertical': vertical,
                'horizontal': horizontal,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'weight': weight,
                'gradOffsetX': gradOffsetX
            }))(
                grid=tuple([int((nx + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[nx, gradOutput.data_ptr(), input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), offset_x.data_ptr(),
                      offset_y.data_ptr(), weight.data_ptr(), gradOffsetX.data_ptr()],
                stream=Stream
            )

            ny = gradOffsetY.nelement()
            cupy_launch('kernel_Kernel_updateGradOffsetY', cupy_kernel('kernel_Kernel_updateGradOffsetY', {
                'gradLoss': gradOutput,
                'input': input,
                'vertical': vertical,
                'horizontal': horizontal,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'weight': weight,
                'gradOffsetX': gradOffsetY
            }))(
                grid=tuple([int((ny + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[ny, gradOutput.data_ptr(), input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(),
                      offset_x.data_ptr(),
                      offset_y.data_ptr(), weight.data_ptr(), gradOffsetY.data_ptr()],
                stream=Stream
            )

            nm = gradWeight.nelement()
            cupy_launch('kernel_Kernel_updateGradWeight', cupy_kernel('kernel_Kernel_updateGradWeight', {
                'gradLoss': gradOutput,
                'input': input,
                'vertical': vertical,
                'horizontal': horizontal,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'gradWeight': gradWeight
            }))(
                grid=tuple([int((nm + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[nm, gradOutput.data_ptr(), input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(),
                      offset_x.data_ptr(),
                      offset_y.data_ptr(), gradWeight.data_ptr()],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradInput, gradVertical, gradHorizontal, gradOffsetX, gradOffsetY, gradWeight, None

