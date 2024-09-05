#include <stdio.h>     //  fopen(), fclose()
#include <stdlib.h>    // mlloc(), free()
#include <pthread.h>
#include <sys/types.h> // int32_t, u_int32_t
#include <sys/time.h>  // gettimeofday()
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#define hash_mult 1.4

//definitions
#define nulltag 89998
#define INITIAL_CAPACITY 10000 //because real-loc is not allowed, this must be big enough to store all entries 

typedef struct entry_t{
	int value; 
	int * coord;
} entry_t; 

typedef struct T_t {
	int order;
	int *shape;
	int entry_count; 
	entry_t *entries;
} T_t;

typedef struct range{
	int start;
	int end; 
	int expanded_size; 
} range;

typedef struct kv{
	int key; 
	int value;
} kv;

typedef struct LNSN{
	int LN; 
	int SN; 
} LNSN;

typedef struct pack{
	int shape; 
	int KSN;
	int sorttag;
	int value;
} pack;

typedef struct Xred{
	int value;
	int KSN;                  
	int shape;
} Xred;

typedef struct Yred{
	int value;
	int shape;
} Yred;

typedef struct Xtype{
	int shape;
	int FSN; 
	int KSN;
	int value; 
} Xtype; 

void tensor_read_new(const char * filename, int * meta_array, int * coord_array, int * value_array);
void print_entries(int * coords, int * values, int order, int count);

void print_array_d(int * array, int size){
	for(int i = 0; i < size; i++){
		printf("%d,", array[i]);
	} printf("\n");
}

void print_array_f(float * array, int size){
	for(int i = 0; i < size; i++){
		printf("%f,", array[i]);
	} printf("\n");
}

//ALLCATING GLOBAL VARIABLES
#define MAX_INPUT_ENTRY_COUNT 1000
#define MAX_HT_SIZE 1000

//CONTRACT FUNCTION 
int contract(int CMODES, int * X_CMODES, int * Y_CMODES, int * X_METADATA, int * X_COORDS, float * X_VALUES, int * Y_METADATA, int * Y_COORDS, float * Y_VALUES, int * Z_METADATA, int * Z_COORDS, float * Z_VALUES) {
	//MEMORY ALLOCATIONS 
	int * FMODES_X 			= (int *)malloc(10 * sizeof(int));
	int * FMODES_Y 			= (int *)malloc(10 * sizeof(int));
	int * Ztmp_value 		= (int *)malloc(INITIAL_CAPACITY * sizeof(int));
	int * Ztmp_coords 		= (int *)malloc(INITIAL_CAPACITY * 5 * sizeof(int)); //Zorder is 5 here
	int * contracting       = (int *)malloc(10 * sizeof(int));
	int * contracting_shape = (int *)malloc(10 * sizeof(int));
	int * acc 				= (int *)malloc(10 * sizeof(int));
	int * Zacc 				= (int *)malloc(10 * sizeof(int));
	int * Z_shape 			= (int *)malloc(10 * sizeof(int));
	int * HT_tag  			= (int *)malloc(MAX_HT_SIZE * sizeof(int));
	int * HT_value 			= (int *)malloc(MAX_HT_SIZE * sizeof(int)); 
	int * curr_entry_coord 	= (int *)malloc(10 * sizeof(int));

	kv * my_ht 			   = (kv *)malloc(MAX_INPUT_ENTRY_COUNT * hash_mult * sizeof(kv));  
	LNSN * X_ht 		 = (LNSN *)malloc(MAX_INPUT_ENTRY_COUNT * hash_mult * sizeof(LNSN));  
	Yred * Ycont 		 = (Yred *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(Yred));
	Xtype * Xunsorted   = (Xtype *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(Xtype));
	pack * Ypacks  		 = (pack *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(pack));
	Xred * Xcont   		 = (Xred *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(Xred));
	Xred * Xacc   		 = (Xred *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(Xred));
	Xred * Xnacc  		 = (Xred *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(Xred));
	range * ranges 		= (range *)malloc(MAX_INPUT_ENTRY_COUNT/2 * sizeof(range));
	int * Xptrs 		  = (int *)calloc(MAX_INPUT_ENTRY_COUNT, sizeof(int));
	int * Xoperations 	  = (int *)malloc(MAX_INPUT_ENTRY_COUNT * sizeof(int)); 
	int * Yptrs 		  = (int *)calloc(MAX_INPUT_ENTRY_COUNT, sizeof(int)); 
	int * Ycounts 		  = (int *)calloc(MAX_INPUT_ENTRY_COUNT, sizeof(int));
	int * removed     	  = (int *)calloc(MAX_INPUT_ENTRY_COUNT, sizeof(int));

	//VARIABLE ALLOCATIONS 
	int XSIZE 				= 0; 
	int HTSIZE 				= 0; 
	int Xcounter 			= 0; 
	int counter 			= 0; 
	int max_range 			= 0; 
	int Xacc_write_ptr 		= 0; 
	int Xnacc_count 		= 0; 
	int Xnacc_write_ptr 	= 0; 
	int range_ptr 			= 0; 
	int Z_entry_count 		= 0;
	int Yfiltered_count  	= 0; 
	int ranges_write_ptr 	= 0;


	////////////////////////////////////////////////////////////////////////////////////////
	//
	//							PHASE I: PRELIMINARY CALCULATIONS
	//
	////////////////////////////////////////////////////////////////////////////////////////
	//EXTRACTING METADATA FROM INPUT TENSORS
	int X_order = X_METADATA[0];
	int X_entry_count = X_METADATA[X_order + 1];
	int Y_order = Y_METADATA[0];
	int Y_entry_count = Y_METADATA[Y_order + 1];

	int XFMODES = X_order - CMODES; 
	int YFMODES = Y_order - CMODES; 


	int wtr = 0; 
	for(int i = 0; i < X_order; i++){
		int in_arr = 0; 
		for (int j = 0; j < CMODES; j++) { if (X_CMODES[j] == i) {in_arr = 1;} }
		if(!in_arr){ FMODES_X[wtr++] = i; }
	}

	wtr = 0; 
	for(int i = 0; i < Y_order; i++){
		int in_arr = 0; 
		for (int j = 0; j < CMODES; j++) { if (Y_CMODES[j] == i) {in_arr = 1;} }
		if(!in_arr){FMODES_Y[wtr++] = i;}
	}

	for(int Ci = 0; Ci < CMODES; Ci++){ contracting_shape[Ci] = X_METADATA[X_CMODES[Ci] + 1]; }

	int Xfree[XFMODES]; int inc = 0; 
	for(int i = 0; i < X_order; ++i){ 
		int in_arr = 0; 
		for (int j = 0; j < CMODES; j++) { if (X_CMODES[j] == i) {in_arr = 1;} }
		if(!in_arr){ Xfree[inc++] = X_METADATA[i+1]; }
	}

	int Yfree[YFMODES]; inc = 0; 
	for(int i = 0; i < Y_order; ++i){ 
		int in_arr = 0; 
		for (int j = 0; j < CMODES; j++) { if (Y_CMODES[j] == i) {in_arr = 1;} }
		if(!in_arr){ Yfree[inc++] = Y_METADATA[i+1];} }

	//CREATING METADATA FOR OUTPUT TENSOR
	int Z_order = X_order + Y_order - 2*CMODES;

	inc = 0; 
	for(int i = 0; i < XFMODES; ++i){Z_shape[inc++] = Xfree[i]; }
	for(int i = 0; i < YFMODES; ++i){Z_shape[inc++] = Yfree[i]; }

	Zacc[0] = 1; for(int i = 1; i < Z_order; i++){ Zacc[i] = Zacc[i-1] * Z_shape[i-1]; }
	//FOR INPUT PROCESSING ON X AND Y CONTRACTING MODES
	HTSIZE = hash_mult * X_entry_count;
	
	for(int i = 0; i < HTSIZE; i++){ my_ht[i].key = -1; }

	//FOR INPUT PROCESSING ON X FREE MODES
	XSIZE = hash_mult * X_entry_count; 
	
	for(int i = 0; i < XSIZE; i++){ X_ht[i].LN = -1; }

	//PERMUTATION 
	//SN CONTRACTING INDICES OF X
	for(int i = 0; i < X_entry_count; ++i){ 
		
		for(int j = 0; j < X_order; j++){  curr_entry_coord[j] = X_COORDS[i*X_order + j]; }
		Xtype curr_pack; 

		//VALUE
		curr_pack.value = X_VALUES[i];

		//SHAPE
		curr_pack.shape = 0; 
		for(int x = 0; x < XFMODES; ++x){ curr_pack.shape += curr_entry_coord[FMODES_X[x]] * Zacc[x]; }

		//curr_pack.FSN = X_setget(curr_pack.shape, XSIZE, X_ht, &Xcounter); 
		//X_setget behavior
			curr_pack.FSN = nulltag;
			unsigned int slot = curr_pack.shape%XSIZE;
			while(X_ht[slot].LN != -1){
				if(X_ht[slot].LN == curr_pack.shape){ curr_pack.FSN = X_ht[slot].SN; break;}
				slot = (slot + 1) % XSIZE;
			}

			if(curr_pack.FSN == nulltag){
				Xcounter = Xcounter + 1;
				LNSN new_lnsn; new_lnsn.LN = curr_pack.shape; new_lnsn.SN = Xcounter-1;
				X_ht[slot] = new_lnsn;

				curr_pack.FSN = (Xcounter-1);
			}
		
			
		//KSN 
		for(int Ci = 0; Ci < CMODES; ++Ci){ contracting[Ci] = curr_entry_coord[X_CMODES[Ci]]; }
		
		//LN BEHAVIOR
		acc[0] = 1;
		for(int i = 1; i < CMODES; i++){
			acc[i] = acc[i-1] * contracting_shape[i-1];
		}
		int mac = 0; 
		for(int i = 0; i < CMODES; i++){
			mac += acc[i] * contracting[i];
		}
		
		//curr_pack.KSN = ht_setget(mac, &counter, HTSIZE, my_ht);
		curr_pack.KSN = nulltag; 
		//ht_setget behavior 
			slot = mac%HTSIZE;

			while(my_ht[slot].key != -1){
				if(my_ht[slot].key == mac){ curr_pack.KSN = my_ht[slot].value; break;}
				slot = (slot + 1) % HTSIZE;
			}

			if(curr_pack.KSN == nulltag){
				counter = counter + 1;
				kv new_kv; new_kv.key = mac; new_kv.value = (counter-1);
				my_ht[slot] = new_kv;
				curr_pack.KSN = (counter-1);
			}
			

		Xunsorted[i] = curr_pack;
	}

	for(int i = 0; i < Y_entry_count; ++i){
		for(int j = 0; j < X_order; j++){ curr_entry_coord[j] = X_COORDS[i*X_order + j]; }
		
		pack curr_pack; 
		
		//VALUE
		curr_pack.value = Y_VALUES[i];
		for(int Ci = 0; Ci < CMODES; ++Ci){ contracting[Ci] = curr_entry_coord[Y_CMODES[Ci]]; }

		//KSN
		//LN BEHAVIOR
		acc[0] = 1;
		for(int i = 1; i < CMODES; i++){
			acc[i] = acc[i-1] * contracting_shape[i-1];
		}
		int mac = 0; 
		for(int i = 0; i < CMODES; i++){
			mac += acc[i] * contracting[i];
		}

		//curr_pack.KSN = ht_get(mac, HTSIZE, my_ht);
		//ht_get behavior
			curr_pack.KSN = nulltag;
			unsigned int slot = mac%HTSIZE;
			while(my_ht[slot].key != -1){
				if(my_ht[slot].key == mac){ curr_pack.KSN = my_ht[slot].value; break;}
				slot = (slot + 1) % HTSIZE;
			}

			if(curr_pack.KSN == nulltag){
				curr_pack.KSN = -1;
			}
			


		if(curr_pack.KSN == -1) {
			curr_pack.sorttag = nulltag;
		} else {
			curr_pack.sorttag = curr_pack.KSN;
			++Yfiltered_count; 
		}

		//SHAPE
		curr_pack.shape = 0; 
		for(int y = 0; y < YFMODES; y++){ curr_pack.shape += curr_entry_coord[FMODES_Y[y]] * Zacc[y+XFMODES]; }

		Ypacks[i] = curr_pack;
	} //for i in Yentry_count
	
	
	//COUNTING X 
	for(int i = 0; i < X_entry_count; ++i){ Xptrs[Xunsorted[i].FSN]++; }

	//ACCUMULATING X 
	for(int i = 1; i < Xcounter; ++i){ Xptrs[i] += Xptrs[i-1]; }
	Xptrs[Xcounter] = Xptrs[Xcounter-1];

	//SORTING X  
	for(int i = 0; i < X_entry_count; ++i){
		Xtype curr_Xpack = Xunsorted[i]; 
		Xred curr_Xred; 
		curr_Xred.value = curr_Xpack.value;
		curr_Xred.KSN = curr_Xpack.KSN;
		curr_Xred.shape = curr_Xpack.shape; 
		Xcont[--Xptrs[Xunsorted[i].FSN]] = curr_Xred;
	}

	//COUNTING Y 
	for(int i = 0; i < Y_entry_count; i++){
		if(Ypacks[i].sorttag != nulltag){ Ycounts[Ypacks[i].sorttag]++; }
	}

	//ACCUMULATING Y 
	Yptrs[0] = Ycounts[0];
	for(int i = 1; i < counter; i++){ Yptrs[i] = Yptrs[i-1] + Ycounts[i]; }
	Yptrs[counter] = Yptrs[counter-1];
	

	//SORTING Y 
	for(int i = 0; i < Y_entry_count; i++){ 
		if(Ypacks[i].sorttag != nulltag){ 
			pack curr_Ypack = Ypacks[i];
			Yred curr_Yred;

			curr_Yred.value = curr_Ypack.value;
			curr_Yred.shape = curr_Ypack.shape;
			Ycont[--Yptrs[Ypacks[i].sorttag]] = curr_Yred;
		}
	}

	//GETTING X OPERATIONS
	for(int i = 0; i < X_entry_count; ++i){ Xoperations[i] = Ycounts[Xcont[i].KSN];}
	
	//FINDING DUPLICATE RANGES 
	int observing = 0; 
	while(observing < X_entry_count-1){
		if(Xcont[observing].shape == Xcont[observing+1].shape){
			//start
			range new_range; 
			new_range.start = range_ptr;
			new_range.expanded_size = 0;  

			//adding entries
			Xacc[Xacc_write_ptr++] = Xcont[observing]; new_range.expanded_size += Xoperations[observing]; removed[observing] = 1;
			observing++; range_ptr++;
			while(observing < (X_entry_count-1) && (Xcont[observing].shape == Xcont[observing+1].shape)){ 
				Xacc[Xacc_write_ptr++] = Xcont[observing]; new_range.expanded_size += Xoperations[observing]; removed[observing] = 1;
				observing++; range_ptr++;
			}
			Xacc[Xacc_write_ptr++] = Xcont[observing]; new_range.expanded_size += Xoperations[observing]; removed[observing] = 1;

			//end
			new_range.end = ++range_ptr; // to make the end not included
			ranges[ranges_write_ptr++] = new_range; 
		} else{ observing++; }
	}
	
	//DIVIDING UP RANGES AND ENTRIES
	Xnacc_count = X_entry_count - Xacc_write_ptr;

	//GROUPING NON-DUPLICATES 
	for(int i = 0; i < X_entry_count; ++i){
		if(!removed[i]){
			Xnacc[Xnacc_write_ptr++] = Xcont[i];
		}
	}

	
	//PHASE 2: 
	for(int i = 0; i < ranges_write_ptr; ++i){
		if(max_range < ranges[i].expanded_size){
			max_range = ranges[i].expanded_size;
		}
	} 
	
	
	for(int I = 0; I < ranges_write_ptr; ++I){
		
		int hashtablesize = ranges[I].expanded_size * hash_mult;
		//defining space by clearing table
		for(int i = 0; i < hashtablesize; i++){ HT_tag[i] = nulltag; HT_value[i] = 0; }

		//perform contraction with hash placing
		for(int i = ranges[I].start; i < ranges[I].end; ++i){
			Xred X_pack = Xacc[i];

			int KSNstart = Yptrs[X_pack.KSN];
			int KSNend = Yptrs[X_pack.KSN+1];

			//DOING CONTRACTION 
			for(int j = KSNstart; j < KSNend; ++j){
				Yred Y_pack = Ycont[j];

				int value = X_pack.value * Y_pack.value;
				int tag = X_pack.shape + Y_pack.shape; 
				
				//hash placing
				int slot = tag % hashtablesize;
				while(HT_tag[slot] != nulltag){
					if(HT_tag[slot] == tag){
						HT_value[slot] += value;
						goto end_of_loop;
					}
					slot = (slot + 1)%hashtablesize;
				}
				HT_tag[slot] = tag;
				HT_value[slot] = value;
				end_of_loop: tag = tag;
			}
		}

		//adding elements to dynamic array
		for(int i = 0; i < hashtablesize; ++i){
			if(HT_tag[i] != nulltag){
				Ztmp_value[Z_entry_count] = HT_value[i];
				
				int remainder = HT_tag[i];

				for(int j = Z_order-1; j > -1; --j){
					int coord = remainder/Zacc[j];
					Ztmp_coords[Z_entry_count * Z_order + j] = coord; 
					remainder -= coord * Zacc[j];
				};
				++Z_entry_count;
			}
		}
	};

	
	//PHASE 1: 
	for(int i = 0; i < Xnacc_count; ++i){
		Xred X_pack = Xnacc[i];

		int contracting_index = X_pack.KSN;

		int KSNstart = Yptrs[contracting_index];
		int KSNend = Yptrs[contracting_index+1];

		//DOING CONTRACTION 
		for(int j = KSNstart; j < KSNend; ++j){
			Yred Y_pack = Ycont[j];

			Ztmp_value[Z_entry_count] = X_pack.value * Y_pack.value;
			int remainder = X_pack.shape + Y_pack.shape;

			for(int j = Z_order-1; j > -1; --j){
				int coord = remainder/Zacc[j];
				remainder -= (coord) * Zacc[j];
				Ztmp_coords[Z_entry_count * Z_order + j] = coord; 
			};

			++Z_entry_count; 
		}
	}	
	
	//EXPORTING TENSOR METADATA
	Z_METADATA[0] = Z_order;
	for(int i = 0; i < Z_order; i++){ Z_METADATA[i+1] = Z_shape[i]; }
	Z_METADATA[Z_order + 1] = Z_entry_count;


	//EXPORTING OUTPUT ENTRIES
	for(int i = 0; i < Z_entry_count; ++i){
		Z_VALUES[i] = Ztmp_value[i];
	}
	for(int j = Z_order-1; j > -1; --j){
		for(int i = 0; i < Z_entry_count; ++i){
			Z_COORDS[j + i*Z_order] = Ztmp_coords[Z_order * i + j];
		}	
	};

	//CONFIRMING RESULTS ARE AS EXPECTED
	int check = 0; 
	for(int i = 0; i < Z_entry_count; ++i){	
		for(int j = Z_order-1; j > -1; --j){
			check += i*Ztmp_coords[Z_order * i + j];
		}
		check += i*Ztmp_value[i]; 
	}
	for(int i = 0; i < Z_order+2; i++){
		check += 4 * i * Z_METADATA[i];
	}


	free(FMODES_X 			);
	free(FMODES_Y 			);
	free(removed     		);
	free(Ztmp_value 		);
	free(Ztmp_coords 		);
	free(contracting      	);
	free(contracting_shape	);
	free(acc 				);
	free(Xptrs 				);
	free(Xoperations 		);
	free(Yptrs 				);
	free(Ycounts 			);
	free(Zacc 				);
	free(Z_shape 			);
	free(HT_tag  			);
	free(HT_value 			);
	free(curr_entry_coord 	);


	free(ranges);
	free(Xacc);
	free(Xnacc);
	free(Ycont);
	free(Xunsorted);
	free(Ypacks);	
	free(my_ht); 	
	free(X_ht);  	
	free(Xcont);
	return check;
}

//main
int main(int argc, char ** argv) {
	int XENTRYCOUNT = 1000; 
	int XORDER = 3; 
	int ZENTRYCOUNT = 1000;
	int ZORDER = 2; 

	int CMODES = 2;

	int * X_CMODES   = (int*)malloc(CMODES * sizeof(int));
	int * Y_CMODES   = (int*)malloc(CMODES * sizeof(int));

	int * X_METADATA = (int*)malloc((XORDER + 2) * sizeof(int));
	int * X_COORDS   = (int*)malloc(XORDER * XENTRYCOUNT * sizeof(int));
	float * X_VALUES = (float*)malloc(XENTRYCOUNT * sizeof(float));

	int * Y_METADATA = (int*)malloc((XORDER + 2) * sizeof(int));
	int * Y_COORDS   = (int*)malloc(XORDER * XENTRYCOUNT * sizeof(int));
	float * Y_VALUES = (float*)malloc(XENTRYCOUNT * sizeof(float));

	int * Z_METADATA = (int*)malloc(10 * sizeof(int));
	int * Z_COORDS   = (int *)malloc(ZORDER * ZENTRYCOUNT * sizeof(int));
	float * Z_VALUES = (float *)malloc(ZENTRYCOUNT * sizeof(float));

	memset(Z_COORDS, 0, ZORDER * ZENTRYCOUNT * sizeof(int));
	memset(Z_VALUES, 0, ZENTRYCOUNT * sizeof(int));

	//CONTRACTING TENSOR FILES
	int outp = contract(CMODES, X_CMODES, Y_CMODES, X_METADATA, X_COORDS, X_VALUES,Y_METADATA,Y_COORDS,Y_VALUES,Z_METADATA, Z_COORDS, Z_VALUES);

	printf("output: %d\n",  outp);

	
	
	
	return 0;
}


















void print_entries(int * coords, int * values, int order, int count){
	printf("Entries: \n");
	for(int e = 0; e < count; e++){
		printf("\t\t(");
		for(int o = 0; o < order; o++){ printf(" %d",coords[e*order + o]);}
		printf(")\t -> %d \n",values[e]);
	}
	
	printf("\n");
}

void tensor_read_new(const char * filename, int * meta_array, int * coord_array, int * value_array) {
	FILE * file_handle = fopen(filename, "r");
	printf("file handle opened\n");
	if (file_handle == NULL) { perror("couldn't open source file"); exit(1); }
	T_t * new_tensor = malloc(sizeof(T_t));
	if (new_tensor == NULL) { printf("failed to allcate memory\n"); exit(1); }

	int meta_pointer = 0; 
	int coord_pointer = 0; 
	int value_pointer = 0; 

	char buffer[4];
	char element_buffer[8];
	int read;
	//first 5 are not needed
	for(int i = 0; i < 5; i++){
		read = fread(buffer, sizeof(buffer), 1, file_handle);
	}
	
	read = fread(buffer, sizeof(buffer), 1, file_handle);
	meta_array[meta_pointer] = *((int*)buffer); int order = meta_array[meta_pointer]; meta_pointer++; //order


	for(int i = 0; i < meta_array[0]; i++){
		read = fread(buffer, sizeof(buffer), 1, file_handle);
		meta_array[meta_pointer] = *((int*)buffer); meta_pointer++; //shape
	}

	read = fread(buffer, sizeof(buffer), 1, file_handle);
	meta_array[meta_pointer] = *((int*)buffer); int entry_count = meta_array[meta_pointer]; meta_pointer++; //entry count

	for(int dim = 0; dim < order; dim++){
	for(int i = 0; i < entry_count; i++){
		read = fread(buffer, sizeof(buffer), 1, file_handle);
		coord_array[i*order + dim] = *((int*)buffer);
	}
	}

	for(int i = 0; i < entry_count; i++){
		read = fread(element_buffer, sizeof(element_buffer), 1, file_handle);
		value_array[i] = 1; if(i == entry_count-1){value_array[i] = 10;}
	}
	printf("new read: %d\n",read);
	fclose(file_handle);
	printf("file handle closed\n");

}

