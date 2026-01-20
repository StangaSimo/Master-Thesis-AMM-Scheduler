
#define TYPE_A 1
#define TYPE_B 0
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "ap_int.h"

extern "C"
{

	struct Registers
	{
		float r[32];
		uint16_t im; /* 16 bit */
		bool c : 1;
		bool i : 1;
		int32_t pc;
	};

	struct Memory
	{
		int8_t instr[32][4];
		float data[1024];
		int32_t size;
	};

	struct Instruction
	{
		int8_t type : 1; /* 1 type A, 0 type B */
		int8_t opcode : 6;
		int8_t rd : 5;
		int8_t ra : 5;
		int8_t rb : 5;
		float im; /* had to remove the bit field */
	};

	float add_Check_Overflow(float a, float b, bool *c)
	{
		double res = (double)a + (double)b; /* only last 32 bit*/
		if (res > INT32_MAX)
		{
			*c = true;
			res = INT32_MAX;
		}
		else if (res < INT32_MIN) /* underflow */
		{

			*c = true;
			res = INT32_MIN;
		}
		else
		{
			*c = false;
		}

		return (float)res;
	}

	uint32_t f_to_u(float n)
	{
		return *(uint32_t *)&n;
	}

	int32_t f_to_i(float n)
	{
		return *(int32_t *)&n;
	}

	float u_to_f(uint32_t n)
	{
		return *(float *)&n;
	}

	void update_PC(struct Registers *reg, int32_t n, bool delay)
	{
		if (!delay) /* Note: if the instruction is in a delay slot, it shouldn't modify the pc reg*/
			reg->pc = reg->pc + n / 4;
	}

	int8_t conv_reg(int8_t n)
	{
		return n & 0b00011111;
	}

	struct Instruction *parse_instruction(int8_t *instr, int8_t type, struct Instruction *res, uint16_t *im)
	{
		if (type) /* Type A */
		{
			res->type = type;
			res->rd = (instr[0] << 3) + ((instr[1] >> 5) & 0b00000111);
			res->ra = instr[1] & 0b00011111;
			res->rb = (instr[2] >> 3) & 0b00011111;
		}
		else /* Type B */
		{
			res->type = type;
			res->rd = ((instr[0] << 3) & 0b00011000) + ((instr[1] >> 5) & 0b00000111);
			res->ra = instr[1] & 0b00011111;
			int16_t n = instr[2];
			n = (n << 8) + (((int16_t)instr[3]) & 0b0000000011111111);
			
			/* TODO */

			if (*im) /* imm istruction before */
			{
				uint32_t im_temp = ((uint32_t)*im << 16) + ((uint32_t)n & 0b00000000000000001111111111111111);
				res->im = *(float *)&im_temp;
				*im = 0;
			}
			else
			{
				int32_t n_32 = (int32_t)n;
				res->im = *(float *)&n_32; 
			}
		}
		return res;
	}

	void run_instruction(int8_t *instruction, struct Memory *data, struct Registers *reg, int8_t instructions[32][4], bool delay)
	{
		struct Instruction instr_data;
		struct Instruction *instr = &instr_data;
		bool carry = false; // carry
		int8_t op_code = (instruction[0] >> 2) & 0b00111111; /* remove last 11 bitfield is not working for the == operand*/
		instr->opcode = (instruction[0] >> 2) & 0b00111111;
		int32_t delayed_instruction, addr;
		int8_t branch_type, is_delayed, is_absolute, is_link;

		switch (op_code)
		{
		case 0x0:
		{ /* ADD 000000 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->ra], reg->r[instr->rb], &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0x4:
		{ /* ADDK 000100 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->ra], reg->r[instr->rb], &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0x2:
		{ /* ADDC 000010 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->ra], reg->r[instr->rb] + reg->c, &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0x6:
		{ /* ADDCK 000110 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->ra], reg->r[instr->rb] + reg->c, &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0x8:
		{ /* ADDI 001000 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, reg->r[instr->ra], &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0xA:
		{ /* ADDIC 001010 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, reg->r[instr->ra] + reg->c, &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0xC:
		{ /* ADDIK 001100 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, reg->r[instr->ra], &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0xE:
		{ /* ADDICK 001110 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, reg->r[instr->ra] + reg->c, &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0x1:
		{ /* RSUB 000001 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->rb], - reg->r[instr->ra], &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0x3:
		{ /* RSUBC 000011 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->rb], - reg->r[instr->ra] + reg->c, &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0x7:
		{ /* RSUBCK 000111 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->rb], - reg->r[instr->ra] + reg->c, &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0x9:
		{ /* RSUBI 001001 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, - reg->r[instr->ra], &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0xB:
		{ /* RSUBIC 001011 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, - reg->r[instr->ra] + reg->c, &carry);
			update_PC(reg, 4, delay);
			break;
		}

		case 0xD:
		{ /* RSUBIK 001101 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, - reg->r[instr->ra], &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0xF:
		{ /* RSUBIKC 001111 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = add_Check_Overflow(instr->im, - reg->r[instr->ra] + reg->c, &carry);
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0x5:
		{ /* RSUBK CMP 000101 */

			/* last 8 bit 00000011  then cmpu */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);

			if (instruction[3] == 0x0)
			{
				reg->r[instr->rd] = add_Check_Overflow(reg->r[instr->rb], - reg->r[instr->ra], &carry);
				reg->c = carry;
			}

			update_PC(reg, 4, delay);
			break;
		}
		case 0x10:
		{ /* MUL 010000*/
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = reg->r[instr->ra] * reg->r[instr->rb];
			update_PC(reg, 4, delay);
			break;
		}

		case 0x24:
		{ /* SRA 100100 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = u_to_f((f_to_u(reg->r[instr->ra]) >> 1));
			reg->c = carry;
			update_PC(reg, 4, delay);
			break;
		}

		case 0x20:
		{ /* OR 100000 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) | f_to_u(reg->r[instr->rb]));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x21:
		{ /* AND 100001 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) & f_to_u(reg->r[instr->rb]));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x22:
		{ /* XOR 100010 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) ^ f_to_u(reg->r[instr->rb]));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x23:
		{ /* ANDN 100011 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) & (~f_to_u(reg->r[instr->rb])));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x28:
		{ /* ORI 101000 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) | f_to_u(instr->im));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x29:
		{ /* ANDI 101001 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) & f_to_u(instr->im));
			update_PC(reg, 4, delay);
			break;
		}
		case 0x2A:
		{ /* XORI 101010 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) ^ f_to_u(instr->im));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x2B:
		{ /* ANDNI 101011 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->r[instr->rd] = u_to_f(f_to_u(reg->r[instr->ra]) & (~f_to_u(instr->im)));
			update_PC(reg, 4, delay);
			break;
		}

		case 0x27:
		{ /* BEQ BGE BGT BLE BLT BNE 100111 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			delayed_instruction = reg->pc + 1; /* next istruction for delayed branch*/
			branch_type = conv_reg(instr->rd) & 0b00001111;
			is_delayed = conv_reg(instr->rd) & 0b00010000;

			if (branch_type == 0x0 && reg->r[instr->ra] == 0x0) /* BEQ D0000 */
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);
			else if (branch_type == 0x5 && reg->r[instr->ra] >= 0x0) /* BGE D0101 */
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);
			else if (branch_type == 0x4 && reg->r[instr->ra] > 0x0) /* BGT D0100 */
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);
			else if (branch_type == 0x3 && reg->r[instr->ra] <= 0x0) /* BLE D0011 */
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);
			else if (branch_type == 0x2 && reg->r[instr->ra] < 0x0) /* BLT D0010 */
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);
			else if (branch_type == 0x1 && reg->r[instr->ra] != 0x0) /* BNE D0001 */
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);
			else
				update_PC(reg, 4, delay);

			// if (is_delayed == 0x10) /* delayed slot */
			//	run_instruction(instructions[delayed_instruction], data, reg, instructions, true);
			break;
		}

		case 0x2F:
		{ /*  BEQI BGEI BGTI BLEI BLTI BNEI 101111 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			delayed_instruction = reg->pc + 1; /* next istruction for delayed branch*/
			branch_type = conv_reg(instr->rd) & 0b00001111;
			is_delayed = conv_reg(instr->rd) & 0b00010000;

			if (branch_type == 0x0 && reg->r[instr->ra] == 0x0) /* BEQI D0000 */
				update_PC(reg, f_to_i(instr->im), delay);
			else if (branch_type == 0x5 && reg->r[instr->ra] >= 0x0) /* BGEI D0101 */
				update_PC(reg, f_to_i(instr->im), delay);
			else if (branch_type == 0x4 && reg->r[instr->ra] > 0x0) /* BGTI D0100 */
				update_PC(reg, f_to_i(instr->im), delay);
			else if (branch_type == 0x3 && reg->r[instr->ra] <= 0x0) /* BLEI D0011 */
				update_PC(reg, f_to_i(instr->im), delay);
			else if (branch_type == 0x2 && reg->r[instr->ra] < 0x0) /* BLTI D0010 */
				update_PC(reg, f_to_i(instr->im), delay);
			else if (branch_type == 0x1 && reg->r[instr->ra] != 0x0) /* BNEI D0001 */
				update_PC(reg, f_to_i(instr->im), delay);
			else
				update_PC(reg, 4, delay);

			// if (is_delayed == 0x10) /* delayed slot */
			//	run_instruction(instructions[delayed_instruction], data, reg, instructions, true);
			break;
		}

		case 0x26:
		{ /* BR BRD BRA BRLD BRAD BRALD 101110 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			delayed_instruction = reg->pc + 1; /* next istruction for delayed branch*/

			/* DAL00 */
			is_absolute = conv_reg(instr->ra) & 0b00001000; /* A for branch with absolute PC = rb */
			is_link = conv_reg(instr->ra) & 0b00000100;			/* L for branch and link rd = PC */
			is_delayed = conv_reg(instr->ra) & 0b00010000;

			if (is_link == 0x4)
				reg->r[conv_reg(instr->rd)] = reg->pc * 4;

			if (is_absolute == 0x8)
				reg->pc = (reg->r[conv_reg(instr->rb)]) / 4;
			else
				update_PC(reg, f_to_i(reg->r[instr->rb]), delay);

			break;
		}
		case 0x2E:
		{ /* BRI BRAI BRID BRAID BRLID BRAILD 101110 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			delayed_instruction = reg->pc + 1; /* next istruction for delayed branch*/

			/* DAL00 */
			is_absolute = conv_reg(instr->ra) & 0b00001000; /* A for branch with absolute PC = rb */
			is_link = conv_reg(instr->ra) & 0b00000100;			/* L for branch and link rd = PC */
			is_delayed = conv_reg(instr->ra) & 0b00010000;

			if (is_link == 0x4)
				reg->r[conv_reg(instr->rd)] = reg->pc * 4;

			if (is_absolute == 0x8)
				reg->pc = f_to_i(instr->im) / 4;
			else
				update_PC(reg, f_to_i(instr->im), delay);

			break;
		}

		case 0x32:
		{ /* LW 110010 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			addr = f_to_u(reg->r[instr->ra]) +  f_to_u(reg->r[instr->rb]);
			reg->r[instr->rd] = data->data[addr];
			update_PC(reg, 4, delay);
			break;
		}
		case 0x36:
		{ /* SW 110110 */
			instr = parse_instruction(instruction, TYPE_A, instr, &reg->im);
			addr = f_to_u(reg->r[instr->ra]) + f_to_u(reg->r[instr->rb]);
			data->data[addr] = reg->r[instr->rd];
			update_PC(reg, 4, delay);
			break;
		}

		case 0x3A:
		{ /* LWI 111010 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			addr = f_to_u(reg->r[instr->ra]) + f_to_u(instr->im);
			reg->r[instr->rd] = data->data[addr];
			update_PC(reg, 4, delay);
			break;
		}

		case 0x3E:
		{ /* SWI 111110 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			addr = f_to_u(reg->r[instr->ra]) + f_to_u(instr->im);
			data->data[addr] = reg->r[instr->rd];
			update_PC(reg, 4, delay);
			break;
		}

		case 0x2C:
		{ /* IMM 101100 */
			instr = parse_instruction(instruction, TYPE_B, instr, &reg->im);
			reg->im = (uint16_t)f_to_u(instr->im);
			update_PC(reg, 4, delay);
			break;
		}

		default:
		{
			/* unknown istruction */
			break;
		}
		}
	}

	void vadd(struct Memory *mem, struct Registers *reg, uint32_t *out, ap_uint<32> my_size)
	{
#pragma HLS INTERFACE m_axi port = mem bundle = gmem
#pragma HLS INTERFACE m_axi port = reg bundle = gmem
#pragma HLS INTERFACE m_axi port = out bundle = gmem
#pragma HLS INTERFACE ap_ctrl_hs port = return

		struct Registers reg_copy;
		struct Memory mem_copy;

		for (int i = 0; i < 32; i++)
			reg_copy.r[i] = reg->r[i];

		reg_copy.c = reg->c;
		reg_copy.i = reg->i;
		reg_copy.pc = reg->pc;
		reg_copy.im = reg->im;

		for (int i = 0; i < 1024; i++)
			mem_copy.data[i] = mem->data[i];

		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 4; j++)
				mem_copy.instr[i][j] = mem->instr[i][j];

		struct Registers *reg_copy_pointer = &reg_copy;
		struct Memory *mem_copy_pointer = &mem_copy;

		while (reg_copy_pointer->pc < my_size)
			run_instruction(mem_copy_pointer->instr[reg_copy_pointer->pc], 
												mem_copy_pointer, 
													reg_copy_pointer, 
														mem_copy_pointer->instr, 
															false);

		out[1] = f_to_i(reg_copy_pointer->r[1]);
	}
}
