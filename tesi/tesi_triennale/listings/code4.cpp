#include "xcl2.hpp"
#include <vector>

#define KERNEL_NUMBER 8

extern "C"
{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

	struct Registers
	{
		int32_t r[32];
		int16_t im; /* for imm instruction */
		bool c : 1;
		bool i : 1;
		int32_t pc;
	};

	struct Memory
	{
		int8_t instr[32][4];
		int32_t data[1024];
		int32_t size;
	};

	struct Registers *inizialize_registers(struct Registers *reg)
	{
		for (int i = 0; i < 32; i++)
		{
			reg->r[i] = 0;
		}
		reg->c = false;
		reg->i = false;
		reg->pc = 0;
		reg->im = 0;
		return reg;
	}

	int8_t **get_instructions_from_file(char const *file_name, int32_t *size)
	{
		FILE *file_1 = fopen(file_name, "rb");
		if (file_1 == NULL)
		{
			fprintf(stderr, "Error: cannot open the file_1 '%s'.\n", file_name);
			exit(1);
		}

		int8_t **instr = (int8_t **)malloc(sizeof(int8_t *));

		fseek(file_1, 0, SEEK_SET);
		while (true)
		{
			int8_t *buffer = (int8_t *)malloc(sizeof(int8_t) * 4);
			int num_bytes = fread(buffer, sizeof(int8_t), 4, file_1);
			if (num_bytes > 0)
			{
				*size = *size + 1;
				instr = (int8_t **)realloc(instr, *size * sizeof(int8_t *));
				instr[*size - 1] = buffer;
			}
			else
			{
				break;
			}
		}
		fclose(file_1);
		return instr;
	}
}

void make_result(int32_t *vector_result, int n)
{
	switch (n)
	{
	case 0:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i]+1;
		break;
	case 1:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i]+5;
		break;
		break;
	case 2:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i]-1;
		break;
	case 3:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i]*2;
		break;
	case 4:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i]*45;
		break;
	case 5:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i] & 45;
		break;
	case 6:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i] | 45;
		break;
	case 7:
		for (int i=0; i<15; i++)
			vector_result[i] = vector_result[i] >> 1;
		break;
	
	default:
		break;
	}	
}
/****************************************************************/
void compare(struct Memory *data, int32_t *vector_result, int n)
{
	bool cmp = true;

	for (int i = 0; i<15; i++) 
		if (data->data[i] != vector_result[i])
			cmp = false;

	if (cmp) 
		std::cout << " Kernel: " << n << " Test Passed\n";
	else 
		std::cout << " Kernel: " << n << " Test Failed\n";
}

/****************************************************************/

void print_vector(int n, struct Memory *data, int j, const char * file)
{
	std::cout << " Kernel: " << j << " file: " << file << " vector: ";
	for (int i = 0; i < n; i++)
	{
		std::cout << data->data[i] << " ";
	}
	std::cout << "\n";
}
/****************************************************************/

void free_data(struct Memory *data, struct Memory *data_out, struct Registers *reg)
{
	free(data);
	free(data_out);
	free(reg);
}

/****************************************************************/

void enqueue_task(char const *file,
									cl::CommandQueue *q,
									cl::Kernel krnl,
									cl::Context context,
									struct Memory *data_out,
									int32_t *vector_result)

{
	cl_int err;
	int32_t instr_size = 0;
	int8_t **vector = get_instructions_from_file(file, &instr_size);
	auto mysize = instr_size;
	struct Memory *data = (struct Memory *)malloc(sizeof(struct Memory));
	struct Registers *reg = (struct Registers *)malloc(sizeof(struct Registers));
	reg = inizialize_registers(reg);

	for (int i = 0; i < 1024; i++)
	{
		data->data[i] = 0;
		data_out->data[i] = 0;
	}

	for (int i = 0; i < 15; i++)
	{
		int32_t n = rand() % 1000;
		data->data[i] = n;
		vector_result[i] = n;
	}

	for (int i = 0; i < mysize; i++)
	{
		data->instr[i][0] = vector[i][0];
		data->instr[i][1] = vector[i][1];
		data->instr[i][2] = vector[i][2];
		data->instr[i][3] = vector[i][3];
	}

	OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(struct Memory), data_out, &err));

	OCL_CHECK(err, cl::Buffer buffer_data(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(struct Memory), data, &err));

	OCL_CHECK(err, cl::Buffer buffer_reg(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(struct Registers), reg, &err));

	// Set the Kernel Arguments
	OCL_CHECK(err, err = krnl.setArg(0, buffer_data));
	OCL_CHECK(err, err = krnl.setArg(1, buffer_reg));
	OCL_CHECK(err, err = krnl.setArg(2, buffer));
	OCL_CHECK(err, err = krnl.setArg(3, mysize));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q->enqueueMigrateMemObjects({buffer_data, buffer_reg}, 0 /* 0 means from host*/));

	// Launch the Kernel
	OCL_CHECK(err, err = q->enqueueTask(krnl));

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q->enqueueMigrateMemObjects({buffer}, CL_MIGRATE_MEM_OBJECT_HOST));
}

/****************************************************************/

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string binaryFile = argv[1];
	cl_int err;
	cl::CommandQueue q;
	cl::Context context;
	cl::Kernel krnl[KERNEL_NUMBER];

	// OPENCL HOST CODE AREA START
	// Create Program and Kernel
	auto devices = xcl::get_xil_devices();

	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file_1 buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;

	for (unsigned int i = 0; i < devices.size(); i++)
	{
		auto device = devices[i];
		// Creating Context and Command Queue for selected Device
		OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, nullptr, &err);

		if (err != CL_SUCCESS)
		{
			std::cout << "Failed to program device[" << i << "] with xclbin file_1!\n";
		}
		else
		{
			std::cout << "Device[" << i << "]: program successful!\n";

			for (int i = 1; i <= KERNEL_NUMBER; i++)
			{
				std::string var = "vadd:{vadd_" + std::to_string(i) + "}";
				OCL_CHECK(err, krnl[i - 1] = cl::Kernel(program, var.c_str(), &err));
			}

			valid_device = true;
			break;
		}
	}
	if (!valid_device)
	{
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	char const *file[KERNEL_NUMBER] = {
			"gpu_1.text", // +1,
			"gpu_2.text", // +5
			"gpu_3.text", // -1
			"gpu_4.text", // *2
			"gpu_5.text", // *45
			"gpu_6.text", // & 45
			"gpu_7.text", // | 45
			"gpu_8.text", // >> 1
	};

	struct Memory *data_out[KERNEL_NUMBER];
	int vector_result[KERNEL_NUMBER][15];

	for (int i = 0; i < KERNEL_NUMBER; i++)
	{
		data_out[i] = (struct Memory *)malloc(sizeof(struct Memory));
		enqueue_task(file[i], &q, krnl[i], context, data_out[i], vector_result[i]);
		make_result(vector_result[i],i);
	}

	OCL_CHECK(err, err = q.finish());

	std::cout << "\n";

	for (int i = 0; i < KERNEL_NUMBER; i++)
	{
		compare(data_out[i],vector_result[i], i);
		free(data_out[i]);
	}

	return EXIT_SUCCESS;
}

