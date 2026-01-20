#include "xcl2.hpp"
#include <vector>

#define DATA_SIZE 256
#define MAX_INSTR 32

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
		int8_t instr[MAX_INSTR][4];
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
		FILE *file = fopen(file_name, "rb");
		if (file == NULL)
		{
			fprintf(stderr, "Error: cannot open the file '%s'.\n", file_name);
			exit(1);
		}

		int8_t **instr = (int8_t **)malloc(sizeof(int8_t *));

		fseek(file, 0, SEEK_SET);
		while (true)
		{
			int8_t *buffer = (int8_t *)malloc(sizeof(int8_t) * 4);
			int num_bytes = fread(buffer, sizeof(int8_t), 4, file);
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
		fclose(file);
		return instr;
	}
}

int main(int argc, char **argv)
{
	// Verfica numero argomenti
	if (argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

	// Istanziamento Variabili Opencl
	cl_int err;
	cl::CommandQueue q;
	cl::Context context;
	cl::Kernel krnl;
	bool valid_device = false;

	// Allocazione risultato
	int32_t *result = (int32_t*)malloc(sizeof(int32_t));
	*result = 0;

	// Lettura device 
	auto devices = xcl::get_xil_devices();

	// 
	std::string binaryFile = argv[1];
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

	for (unsigned int i = 0; i < devices.size(); i++)
	{
		auto device = devices[i];

		// Creazione della coda ci comando e del contesto per i device presenti
		OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, nullptr, &err);

		if (err != CL_SUCCESS)
		{
			std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
		}
		else
		{
			std::cout << "Device[" << i << "]: program successful!\n";

			//Creazione Kernel
			OCL_CHECK(err, krnl = cl::Kernel(program, "vadd", &err)); 
			valid_device = true;
			break; // Device valido trovato
		}
	}
	if (!valid_device)
	{
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	//char const *file = "add.text";
	// char const* file = "sub.text";
	// char const* file = "cmp.text";
	// char const* file = "data.text";
	// char const* file = "bitop.text";
	// char const* file = "absolute_branch.text";
	char const* file = "branch.text";

	int32_t instr_size = 0;
	int8_t **instr_vector = get_instructions_from_file(argv[2], &instr_size);
	auto mysize = instr_size;

	struct Memory *data = (struct Memory *)malloc(sizeof(struct Memory));
	struct Registers *reg = (struct Registers *)malloc(sizeof(struct Registers));

	// Inizializzazione Registri
	reg = inizialize_registers(reg);

	// Inizializzazione Memoria
	for (int i = 0; i < 1024; i++)
		data->data[i] = 0;

	// Copia istruzioni
	for (int i = 0; i < mysize; i++)
	{
		data->instr[i][0] = instr_vector[i][0];
		data->instr[i][1] = instr_vector[i][1];
		data->instr[i][2] = instr_vector[i][2];
		data->instr[i][3] = instr_vector[i][3];
	}

	 // Allocazione Buffer nella memoria globale della FPGA
	OCL_CHECK(err, cl::Buffer buffer_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(int32_t), result, &err));
	OCL_CHECK(err, cl::Buffer buffer_data(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(struct Memory), data, &err));
	OCL_CHECK(err, cl::Buffer buffer_reg(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(struct Registers), reg, &err));

	// Configurazione degli argomenti del kernel
	OCL_CHECK(err, err = krnl.setArg(0, buffer_data));
	OCL_CHECK(err, err = krnl.setArg(1, buffer_reg));
	OCL_CHECK(err, err = krnl.setArg(2, buffer_out));
	OCL_CHECK(err, err = krnl.setArg(3, mysize));

	// Migrazione dei dati dalla memoria host alla memoria globale della FPGAy
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_data, buffer_reg}, 0 /* 0 means from host*/));

  // Inizio esecuzione kerne
	OCL_CHECK(err, err = q.enqueueTask(krnl));

	// Migrazione risultati dall'accelleratore alla memoria host 
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));

	// Attesa che il kernel completi l'esecuzione
	OCL_CHECK(err, err = q.finish());

	std::cout << "\n\n FPGA RESULT " << *result << "\n\n";

	return EXIT_SUCCESS;
}

