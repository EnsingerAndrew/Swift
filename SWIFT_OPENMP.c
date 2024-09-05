#include <stdio.h>     // printf(), fopen(), fclose()
#include <stdlib.h>    // mlloc(), free()
#include <sys/types.h> // int32_t, u_int32_t
#include <sys/time.h>  // gettimeofday()
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <time.h>

#define _POSIX_C_SOURCE 199309L

#define hash_mult 1.4

//definitions
#define NUM_THREADS 8
#define nulltag ULLONG_MAX
#define INITIAL_CAPACITY 10000

typedef double tensor_element;
typedef u_int32_t pointer_type;
typedef unsigned long long tagtype; 


typedef struct entry_t{
	tensor_element value; 
	pointer_type * coord;
} entry_t; 

typedef struct T_t {
	pointer_type order;
	pointer_type *shape;
	pointer_type entry_count; 
	entry_t *entries;
} T_t;

typedef struct outputT_t {
	pointer_type order;
	pointer_type *shape;
	pointer_type entry_count; 
	pointer_type **indices;
	tensor_element * values; 
} outputT_t;

typedef struct range{
	pointer_type start;
	pointer_type end; 
	pointer_type expanded_size; 
} range;

typedef struct kv{
	tagtype key; 
	int value;
} kv;

typedef struct LNSN{
	int LN; 
	pointer_type SN; 
} LNSN;

typedef struct pack{
	tagtype shape; 
	int KSN;
	tagtype sorttag;
	tensor_element value;
} pack;

typedef struct Xred{
	tensor_element value;
	tagtype KSN;                    // make this a pointer_type
	tagtype shape;
} Xred;

typedef struct Yred{
	tensor_element value;
	tagtype shape;
} Yred;

typedef struct Xtype{
	tagtype shape;
	pointer_type FSN; 
	tagtype KSN;
	tensor_element value; 
} Xtype; 

typedef struct tagged_entry{
	tensor_element value; 
	tagtype tag; 
} tagged_entry;


int max(int a, int b) { return (a > b) ? a : b; }
int min(int a, int b) { return (a < b) ? a : b; }
T_t * tensor_read(const char * filename);
void print_tagged_entry(const tagged_entry ent);
void print_array(const pointer_type arr[], const pointer_type size);
void print_tensor_type(const T_t * input);
tagtype LN(const pointer_type coordinate[], const pointer_type shape[], pointer_type size);
int inArray(pointer_type num, pointer_type arr[], int size);
pointer_type ptr_LN(const pointer_type coordinate[], const pointer_type shape[], pointer_type size);
void tensor_free(T_t * tensor);
T_t * tensor_alloc(pointer_type ORDER, pointer_type ENTRYCOUNT, pointer_type * SHAPE);
outputT_t * tensor_alloc_output(pointer_type ORDER, pointer_type ENTRYCOUNT, pointer_type * SHAPE);
void print_tensor_output(const outputT_t * input); 
void tensor_free_output(outputT_t * tensor);

void print_tag_array(const tagtype arr[], const pointer_type size){
	for(pointer_type i = 0; i < size; i++){
		printf("%lld,", arr[i]);
	}
	printf("\n");
}


void print_range(range r){ printf("Range: [%d, %d), Size: %d \n", r.start, r.end, r.expanded_size); }
void print_Xtype(Xtype x){ printf("Xtype: FSN(%d), KSN(%lld), value(%f), shape(%lld)\n",x.FSN, x.KSN, x.value, x.shape);  }
void print_Xred(Xred X){ printf("Xred: value(%f), KSN(%lld), shape(%lld)\n", X.value, X.KSN, X.shape);}

//DYNAMIC ARRAY IMPLEMENTATION start
typedef struct {
    tagged_entry *array;
    int size;
    int capacity;
} DynamicArray;

DynamicArray* createDynamicArray() {
    DynamicArray *dynArr = malloc(sizeof(DynamicArray));
    dynArr->array = malloc(INITIAL_CAPACITY * sizeof(tagged_entry));
    dynArr->size = 0;
    dynArr->capacity = INITIAL_CAPACITY;
    return dynArr;
}

void appendDynamicArray(DynamicArray *dynArr, tagged_entry element) {
    if (dynArr->size == dynArr->capacity) {
        dynArr->capacity *= 2;
    	dynArr->array = realloc(dynArr->array, dynArr->capacity * sizeof(tagged_entry));
    }
    dynArr->array[dynArr->size++] = element;
}

tagged_entry getDynamicArray(DynamicArray *dynArr, int index) {
    if (index < 0 || index >= dynArr->size) {
        printf("Index out of bounds\n");
        return dynArr->array[0]; // Return a default value indicating error
    }
    return dynArr->array[index];
}

void freeDynamicArray(DynamicArray *dynArr) {
    free(dynArr->array);
    free(dynArr);
}

//DYNAMIC ARRAY IMPLEMENTATION end

int contract(T_t * X, T_t * Y, outputT_t * Z, pointer_type CMODES, pointer_type * CMODES_X, pointer_type * CMODES_Y, int prt) {
	if(prt){
		printf("\nPrinting Tnsr X:\n");
		print_tensor_type(X);
		printf("\nPrinting Tnsr Y:\n");
		print_tensor_type(Y);
	}
	

	struct timespec start, phase; 
	double elapsed_us;

	double preliminary_us;
	double permutation_us;
	double sorting_us;
	double contraction_us; 
	double accumulation_us; 
	double writeback_us;

	//STARTING TIMER TO MEASURE CONTRACTION
	if(prt) printf("Start Timer\n");
	clock_gettime(CLOCK_MONOTONIC, &start);				//starting timer
	////////////////////////////////////////////////////////////////////////////////////////
	//
	//							PHASE I: PRELIMINARY CALCULATIONS
	//
	////////////////////////////////////////////////////////////////////////////////////////
	//EXTRACTING METADATA FROM INPUT TENSORS
	pointer_type X_order = X->order;
	pointer_type X_entry_count = X->entry_count;
	pointer_type Y_order = Y->order;
	pointer_type Y_entry_count = Y->entry_count;

	const pointer_type XFMODES = X_order - CMODES; 
	const pointer_type YFMODES = Y_order - CMODES; 

	pointer_type FMODES_X[XFMODES];
	pointer_type FMODES_Y[YFMODES];

	pointer_type wtr = 0; 
	for(pointer_type i = 0; i < X_order; i++){
		if(!inArray(i,CMODES_X,CMODES)){FMODES_X[wtr++] = i;};
	}

	wtr = 0; 
	for(int i = 0; i < Y_order; i++){
		if(!inArray(i,CMODES_Y, CMODES)){FMODES_Y[wtr++] = i;};
	}

	if(prt) printf("FMODES_X: "); print_array(FMODES_X,XFMODES);
	if(prt) printf("FMODES_Y: "); print_array(FMODES_Y,YFMODES);

	pointer_type contracting_shape[CMODES];
	for(pointer_type Ci = 0; Ci < CMODES; ++Ci){ contracting_shape[Ci] = X->shape[CMODES_X[Ci]]; }

	pointer_type Xfree[XFMODES]; pointer_type inc = 0; 
	for(pointer_type i = 0; i < X_order; ++i){ if(!inArray(i,CMODES_X,CMODES)){ Xfree[inc++] = X->shape[i]; } }

	pointer_type Yfree[YFMODES]; inc = 0; 
	for(pointer_type i = 0; i < Y_order; ++i){ if(!inArray(i,CMODES_Y,CMODES)){ Yfree[inc++] = Y->shape[i];} }

	if(prt) printf("Xshape: "); print_array(X->shape, X_order);
	if(prt) printf("Yshape: "); print_array(Y->shape, Y_order);
	if(prt) printf("contracting_shape: "); print_array(contracting_shape,CMODES);



	//CREATING METADATA FOR OUTPUT TENSOR
	const pointer_type Z_order = X_order + Y_order - 2*CMODES;

	pointer_type Z_shape[Z_order]; inc = 0; 
	for(pointer_type i = 0; i < XFMODES; ++i){Z_shape[inc++] = Xfree[i]; }
	for(pointer_type i = 0; i < YFMODES; ++i){Z_shape[inc++] = Yfree[i]; }

	tagtype Zacc[Z_order];
	Zacc[0] = 1; for(pointer_type i = 1; i < Z_order; i++){ Zacc[i] = Zacc[i-1] * Z_shape[i-1]; }
	


	//ALLOCATING GLOBAL VARIABLES
	//FOR INPUT SORTING
	pointer_type * Xptrs; 

	pointer_type * Ycounts; 
	pointer_type * Yptrs; 

	Xtype * Xunsorted = (Xtype *)malloc(X_entry_count * sizeof(Xtype));
	pack * Ypacks = (pack *)malloc(Y_entry_count * sizeof(pack));

	Xred * Xcont = (Xred *)malloc(X_entry_count * sizeof(Xred));
	Xred * Xacc = (Xred *)malloc(X_entry_count * sizeof(Xred));
	Xred * Xnacc = (Xred *)malloc(X_entry_count * sizeof(Xred));
	
	
	pointer_type Xacc_write_ptr = 0; 
	pointer_type Xnacc_count = 0; 
	pointer_type Xnacc_write_ptr = 0; 
	pointer_type range_ptr = 0; 

	pointer_type * removed = (pointer_type *)calloc(X_entry_count, sizeof(Xred));

	Yred * Ycont;
	pointer_type Yfiltered_count = 0; 

	//FOR DIVIDING WORKLOAD EVENLY

	pointer_type * Xoperations = (pointer_type *)malloc((X_entry_count)*sizeof(pointer_type));


	pointer_type ideal_range_division = 0; 
	pointer_type ideal_entry_division = 0;

	pointer_type range_divisions[NUM_THREADS+1];
	pointer_type entry_divisions[NUM_THREADS+1];

	volatile pointer_type Z_entry_count = 0;

	volatile int writeback_ready = 0; 


	//FOR ACCUMULATION
	range * ranges = (range *)malloc(X_entry_count/2 * sizeof(range));
	pointer_type ranges_write_ptr = 0; 

	//FOR WRITEBACK
	pointer_type output_start_ptrs[NUM_THREADS+1];



	//BARRIERS TO ALIGN THREADS
	volatile int SNXdone = 0; 
	volatile int SNYdone = 0; 
	volatile pointer_type Xsorting_done = 0;
	volatile pointer_type readyToContract = 0; 
	volatile int contraction_done = 0; 

	//FOR DIVIDING ENTRIES AMONG THREADS
	pointer_type SNYdivs[NUM_THREADS+1];
	

	
	
			

	//FOR INPUT PROCESSING ON X AND Y CONTRACTING MODES
	const int HTSIZE = hash_mult * X_entry_count;
	kv * my_ht = (kv *)malloc(HTSIZE * sizeof(kv));    int counter = 0; 
	for(int i = 0; i < HTSIZE; i++){ my_ht[i].key = -1; }

	int ht_setget(const int KEY, int *counter){
		unsigned int slot = KEY%HTSIZE;

		while(my_ht[slot].key != -1){
			if(my_ht[slot].key == KEY){ return my_ht[slot].value; }
			slot = (slot + 1) % HTSIZE;
		}

		*counter = *counter + 1;
		kv new_kv; new_kv.key = KEY; new_kv.value = (*counter-1);
		my_ht[slot] = new_kv;

		return (*counter - 1);
	}

	int ht_get(const int KEY){
		unsigned int slot = KEY%HTSIZE;
		while(my_ht[slot].key != -1){
			if(my_ht[slot].key == KEY){ return my_ht[slot].value; }
			slot = (slot + 1) % HTSIZE;
		}

		return -1;
	}


	//FOR INPUT PROCESSING ON X FREE MODES
	const int XSIZE = hash_mult * X_entry_count; 
	LNSN * X_ht = (LNSN *)malloc(XSIZE * sizeof(LNSN));  int Xcounter = 0; 
	for(int i = 0; i < XSIZE; i++){ X_ht[i].LN = -1; }

	int X_setget(const tagtype KEY){
		unsigned int slot = KEY%XSIZE;

		while(X_ht[slot].LN != -1){
			if(X_ht[slot].LN == KEY){ return X_ht[slot].SN; }
			slot = (slot + 1) % XSIZE;
		}

		//change this later
		Xcounter++;
		LNSN new_lnsn; new_lnsn.LN = KEY; new_lnsn.SN = Xcounter-1;
		X_ht[slot] = new_lnsn;

		return (Xcounter-1);
	}

	// int X_get(const int KEY){
	// 	unsigned int slot = KEY%XSIZE;
	// 	while(X_ht[slot].LN != -1){
	// 		if(X_ht[slot].LN == KEY){ return X_ht[slot].SN; }
	// 		slot = (slot + 1) % XSIZE;
	// 	}

	// 	return -1;
	// }
	//printf("\thash tables initialized\n");



	//PERMUTATION PART 1
	omp_set_num_threads(2);
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section //SN CONTRACTING INDICES OF X
            {
                for(int i = 0; i < X_entry_count; ++i){ 
					entry_t curr_entry = X->entries[i];
					Xtype curr_pack; 

					//VALUE
					curr_pack.value = curr_entry.value;

					//SHAPE
					curr_pack.shape = 0; 
					for(int x = 0; x < XFMODES; ++x){ curr_pack.shape += curr_entry.coord[FMODES_X[x]] * Zacc[x]; }

					curr_pack.FSN = X_setget(curr_pack.shape); //note: X_setget takes pointer_type not tag, this may cause problems

					//KSN 
					pointer_type contracting[CMODES];
					for(pointer_type Ci = 0; Ci < CMODES; ++Ci){ contracting[Ci] = curr_entry.coord[CMODES_X[Ci]]; }
					curr_pack.KSN = ht_setget(LN(contracting, contracting_shape, CMODES), &counter);

					Xunsorted[i] = curr_pack;
				}
				SNXdone++;
            }

            #pragma omp section //CREATING DIVISIONS FOR Y ENTRIES
            {
				pointer_type perfect_division = Y_entry_count/NUM_THREADS;

				SNYdivs[0] = 0;  
				for(int i = 0; i < NUM_THREADS; i++){ SNYdivs[i+1] = SNYdivs[i] + perfect_division; }
				SNYdivs[NUM_THREADS] = Y_entry_count;

				SNXdone++;
            }
        }
    }

	//PERMUTATION PART 2 
	omp_lock_t SNY_lock;
	omp_init_lock(&SNY_lock);

    #pragma omp parallel num_threads(8)
    {
		int idx = omp_get_thread_num();
		//SNY 
		pointer_type mystart = SNYdivs[idx];
		pointer_type myend = SNYdivs[idx+1];

		pointer_type Yfiltered_tmp = 0; 
		pointer_type contracting[CMODES];

		for(int i = mystart; i < myend; ++i){
			entry_t curr_entry = Y->entries[i];
			pack curr_pack; 
			//VALUE
			curr_pack.value = curr_entry.value;
			
			for(pointer_type Ci = 0; Ci < CMODES; ++Ci){ contracting[Ci] = curr_entry.coord[CMODES_Y[Ci]]; }

			//KSN
			curr_pack.KSN = ht_get(LN(contracting, contracting_shape, CMODES));

			if(curr_pack.KSN == -1) {
				curr_pack.sorttag = nulltag;
			} else {
				curr_pack.sorttag = curr_pack.KSN;
				++Yfiltered_tmp; 
			}

			//SHAPE
			curr_pack.shape = 0; 
			for(int y = 0; y < YFMODES; y++){ curr_pack.shape += curr_entry.coord[FMODES_Y[y]] * Zacc[y+XFMODES]; }

			Ypacks[i] = curr_pack;
		}

		omp_set_lock(&SNY_lock);
		SNYdone++; Yfiltered_count += Yfiltered_tmp;
		omp_unset_lock(&SNY_lock);
	}
	omp_destroy_lock(&SNY_lock);




	//INPUT SORTING 
	omp_set_num_threads(2);
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section // SORTING X ENTRIES
            {
               //for(int i = 0; i < 20; i++){ print_Xtype(Xunsorted[i]); }
				Xptrs = (pointer_type *)calloc(Xcounter+1, sizeof(pointer_type));
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

            }

            #pragma omp section // SORTING Y ENTRIES
            {
				Ycounts = (pointer_type *)calloc(counter, sizeof(pointer_type));
				Yptrs = (pointer_type *)calloc(counter+1, sizeof(pointer_type));
				//COUNTING Y 
				for(int i = 0; i < Y_entry_count; i++){
					if(Ypacks[i].sorttag != nulltag){ Ycounts[Ypacks[i].sorttag]++; }
				}

				//ACCUMULATING Y 
				Yptrs[0] = Ycounts[0];
				for(int i = 1; i < counter; i++){ Yptrs[i] = Yptrs[i-1] + Ycounts[i]; }
				Yptrs[counter] = Yptrs[counter-1];

				//SORTING Y 
				Ycont = (Yred *)malloc(Yfiltered_count * sizeof(Yred));

				for(int i = 0; i < Y_entry_count; i++){ 
					if(Ypacks[i].sorttag != nulltag){ 
						pack curr_Ypack = Ypacks[i];
						Yred curr_Yred;

						curr_Yred.value = curr_Ypack.value;
						curr_Yred.shape = curr_Ypack.shape;
						Ycont[--Yptrs[Ypacks[i].sorttag]] = curr_Yred;
					}
				}
            }
        }
    }



	//FINDING DUPLICAES IN X PACKS SORTED	
	if(prt) printf("Xcounter: %d\n", Xcounter);
	//printf("Xptrs:");print_array(Xptrs,20);

	//for(int i = 0; i < 20; i++){ print_Xred(Xcont[i]); }
	//GETTING X OPERATIONS
	for(int i = 0; i < X_entry_count; ++i){ Xoperations[i] = Ycounts[Xcont[i].KSN];}

	//FINDING DUPLICATE RANGES 
	pointer_type observing = 0; 
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
	ideal_entry_division = Xnacc_count/NUM_THREADS;
	ideal_range_division = ranges_write_ptr/NUM_THREADS;

		//DIVIDING UP RANGES
		range_divisions[0] = 0; 
		for(int i = 1; i < NUM_THREADS; i++){
			range_divisions[i] = range_divisions[i-1] + ideal_range_division;
		}
		range_divisions[NUM_THREADS] = ranges_write_ptr;

		//DIVIDING UP ENTRIES
		entry_divisions[0] = 0; 
		for(int i = 1; i < NUM_THREADS; i++){
			entry_divisions[i] = entry_divisions[i-1] + ideal_entry_division;
		}
		entry_divisions[NUM_THREADS] = Xnacc_count;

	//DISPLAYING BOUNDS 
	if(prt) printf("range_divisions: "); print_array(range_divisions, NUM_THREADS+1);
	if(prt) printf("entry_divisions: "); print_array(entry_divisions, NUM_THREADS+1);

	//GROUPING NON-DUPLICATES 
	for(int i = 0; i < X_entry_count; ++i){
		if(!removed[i]){
			Xnacc[Xnacc_write_ptr++] = Xcont[i];
		}

	}

	//FREE UNNEEDED MEMORY
	free(removed);
	free(Xunsorted);
	free(Ycounts); //LAST USED IN SORTING
	free(Xoperations); //SORTING
	free(Ypacks); //SORTING
	free(my_ht); //PERMUTATION
	free(X_ht);  //PERMUTATION
	free(Xcont);

	omp_lock_t contraction_lock;
	omp_init_lock(&contraction_lock);

	//CONTRACTION PART 1
	#pragma omp parallel num_threads(8)
    {
		int idx = omp_get_thread_num();
		tensor_element * Ztmp_value = (tensor_element *)malloc(INITIAL_CAPACITY * sizeof(tensor_element));
		pointer_type * Ztmp_coords = (pointer_type *)malloc(INITIAL_CAPACITY * Z_order * sizeof(pointer_type));

		pointer_type Ztmp_count = 0; 
		pointer_type Ztmp_capacity = INITIAL_CAPACITY; 

		//PHASE 2: 
		pointer_type range_start = range_divisions[idx];
		pointer_type range_end = range_divisions[idx+1];

		pointer_type max_range = 0; 

		for(pointer_type i = range_start; i < range_end; ++i){
			max_range = max(max_range, ranges[i].expanded_size);
		}

		pointer_type max_htsize = max_range * hash_mult;

		//ALLOCATING HASH TABLE 
		tagtype * HT_tag = (tagtype *)malloc(max_htsize * sizeof(tagtype));
		tensor_element * HT_value = (tensor_element *)malloc(max_htsize * sizeof(tensor_element));

		for(pointer_type I = range_start; I < range_end; ++I){
			
			pointer_type hashtablesize = ranges[I].expanded_size * hash_mult;
			//defining space by clearing table
			for(int i = 0; i < hashtablesize; i++){ HT_tag[i] = nulltag; HT_value[i] = 0; }

			//perform contraction with hash placing
			for(pointer_type i = ranges[I].start; i < ranges[I].end; ++i){
				Xred X_pack = Xacc[i];

				pointer_type KSNstart = Yptrs[X_pack.KSN];
				pointer_type KSNend = Yptrs[X_pack.KSN+1];

				//DOING CONTRACTION 
				for(pointer_type j = KSNstart; j < KSNend; ++j){
					Yred Y_pack = Ycont[j];

					tensor_element value = X_pack.value * Y_pack.value;
					tagtype tag = X_pack.shape + Y_pack.shape; 
					
					//hash placing
					pointer_type slot = tag % hashtablesize;
					while(HT_tag[slot] != nulltag){
						if(HT_tag[slot] == tag){
							HT_value[slot] += value;
							goto end_of_loop;
						}
						slot = (slot + 1)%hashtablesize;
					}
					HT_tag[slot] = tag;
					HT_value[slot] = value;
					end_of_loop:
				}
			}

			//adding elements to dynamic array
			for(int i = 0; i < hashtablesize; ++i){
				if(HT_tag[i] != nulltag){
					if (Ztmp_count == Ztmp_capacity) {
						Ztmp_capacity *= 2;

						Ztmp_coords = realloc(Ztmp_coords, Ztmp_capacity * Z_order * sizeof(tagtype));
						Ztmp_value = realloc(Ztmp_value, Ztmp_capacity * sizeof(tensor_element));
					}

					Ztmp_value[Ztmp_count] = HT_value[i];
					
					tagtype remainder = HT_tag[i];

					for(int j = Z_order-1; j > -1; --j){
						pointer_type coord = remainder/Zacc[j];
						Ztmp_coords[Ztmp_count * Z_order + j] = coord; 
						remainder -= coord * Zacc[j];
					};
					++Ztmp_count;
				}
			}
		};

		free(HT_tag);
		free(HT_value);

		//PHASE 1: 
		pointer_type contraction_start = entry_divisions[idx];
		pointer_type contraction_end = entry_divisions[idx+1];

		for(pointer_type i = contraction_start; i < contraction_end; ++i){
			Xred X_pack = Xnacc[i];

			tagtype contracting_index = X_pack.KSN;

			pointer_type KSNstart = Yptrs[contracting_index];
			pointer_type KSNend = Yptrs[contracting_index+1];

			//DOING CONTRACTION 
			for(pointer_type j = KSNstart; j < KSNend; ++j){
				Yred Y_pack = Ycont[j];
				
				if (Ztmp_count == Ztmp_capacity) {
					Ztmp_capacity *= 1.5;

					Ztmp_coords = realloc(Ztmp_coords, Ztmp_capacity * Z_order * sizeof(pointer_type));
					Ztmp_value = realloc(Ztmp_value, Ztmp_capacity * sizeof(tensor_element));
				}

				Ztmp_value[Ztmp_count] = X_pack.value * Y_pack.value;
				tagtype remainder = X_pack.shape + Y_pack.shape;

				for(int j = Z_order-1; j > -1; --j){
					pointer_type coord = remainder/Zacc[j];
					remainder -= (coord) * Zacc[j];
					Ztmp_coords[Ztmp_count * Z_order + j] = coord; 
				};

				++Ztmp_count; 
			}
		}	

		omp_set_lock(&contraction_lock);
		Z_entry_count += Ztmp_count;
		output_start_ptrs[idx+1] = Ztmp_count;
		contraction_done++;
		omp_unset_lock(&contraction_lock);

		#pragma omp barrier

		if(idx == 0){ 
			if(prt) printf("\n\nDONE WITH CONTRACTION PHASE\n"); 
			Z = tensor_alloc_output(Z_order, Z_entry_count, Z_shape);
			output_start_ptrs[0] = 0;
			for(int i = 1; i < NUM_THREADS; i++){
				output_start_ptrs[i+1] += output_start_ptrs[i];
			}
			writeback_ready = 1;
		}

		while(!writeback_ready){}
		if(idx == 0){ if(prt) printf("\n\nSTARTING WRITEBACK PHASE\n"); }
		//TRAVERSE DYNAMIC ARRAY AND ADD TO Z->entries
		pointer_type my_output_ptr = output_start_ptrs[idx];

		for(int i = 0; i < Ztmp_count; ++i){
			Z->values[my_output_ptr++] = Ztmp_value[i];
		}
	

		for(int j = Z_order-1; j > -1; --j){
			my_output_ptr = output_start_ptrs[idx];
			for(int i = 0; i < Ztmp_count; ++i){
				Z->indices[j][my_output_ptr++] = Ztmp_coords[Z_order * i + j];
			}	
		};

		free(Ztmp_value);
		free(Ztmp_coords);
	}
	omp_destroy_lock(&contraction_lock);

	clock_gettime(CLOCK_MONOTONIC, &phase);
	elapsed_us = (phase.tv_sec - start.tv_sec) * 1000000LL + (phase.tv_nsec - start.tv_nsec) / 1000LL;
	if(prt) printf("\n\nFINISHED\t%f us , %d ms \n",elapsed_us, (int)(elapsed_us/1000));
	
	int preliminary_time  = round(preliminary_us );
	int permutation_time  = round(permutation_us );	
	int sorting_time      = round(sorting_us     );
	int contraction_time  = round(contraction_us );
	int writeback_time    = round(writeback_us   );

	if(prt) printf("\n");
	if(prt) printf(" Preliminary: %d us\n", preliminary_time );
	if(prt) printf(" Permutation: %d us\n", permutation_time );
	if(prt) printf("     Sorting: %d us\n", sorting_time     );
	if(prt) printf(" Contraction: %d us\n", contraction_time );
	if(prt) printf("   Writeback: %d us\n", writeback_time   );

	if(prt) printf("\nPrinting Tnsr Z:\n");
	print_tensor_output(Z); 

	if(prt) printf("\nDeallocating memory...\t");

	tensor_free_output(Z);

	free(ranges); //ACCUMULATION
	free(Yptrs); //CONTRCATION
	free(Xacc);
	free(Xnacc);
	free(Ycont);
		
	if(prt) printf("\n");

	return (int)(elapsed_us/1000); 

}

//main
int main(int argc, char ** argv) {


	if (argc != 4) { printf("Usage: %s IN_X IN_Y OUT\n", argv[0]); exit(1); }

	T_t * X = tensor_read(argv[1]); //CHANGE TO tensor_read_alt() FOR iTensor FILES
	T_t * Y = tensor_read(argv[2]); 
	outputT_t * Z; 

	pointer_type CMODES = 2;
	pointer_type CMODES_X[] = {2,3};
	pointer_type CMODES_Y[] = {2,3};

	printf("Execution time: %d\n",contract(X, Y, Z, CMODES, CMODES_X, CMODES_Y,0));
	// for(int i = 0; i < 10; i++){
	// 	printf("Execution time: %d\n",contract(X, Y, Z, CMODES, CMODES_X, CMODES_Y,0));
	// }
	
	
	
	/**/
	return 0;
}





































tagtype LN(const pointer_type coordinate[], const pointer_type shape[], pointer_type size){
	tagtype acc[size];
	acc[0] = 1;
	for(pointer_type i = 1; i < size; i++){
		acc[i] = acc[i-1] * shape[i-1];
	}
	tagtype mac = 0; 
	for(pointer_type i = 0; i < size; i++){
		mac += acc[i] * coordinate[i];
	}
	return mac;
}

pointer_type ptr_LN(const pointer_type coordinate[], const pointer_type shape[], pointer_type size){
	pointer_type acc[size];
	acc[0] = 1;
	for(pointer_type i = 1; i < size; i++){
		acc[i] = acc[i-1] * shape[i-1];
	}
	pointer_type mac = 0; 
	for(pointer_type i = 0; i < size; i++){
		mac += acc[i] * coordinate[i];
	}
	return mac;
}



int inArray(pointer_type num, pointer_type arr[], int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == num) { return 1; }
    }
    return 0;
}




void print_tensor_type(const T_t * input){
	pointer_type O = input->order;
	pointer_type E = input->entry_count;
	printf("\tOrder: %d\n",O);
	printf("\tShape:");
	for(pointer_type i = 0; i < O; i++){ printf("  %d",input->shape[i]); }
	printf("\n\tEntry Count:%d\n", E);
	for(int e = 0; e < 20; e++){
		printf("\t\t(");
		for(pointer_type o = 0; o < O; o++){ printf(" %d",input->entries[e].coord[o]);}
		printf(")\t -> %f \n",input->entries[e].value);
	}
	// printf("\t\t...\n");
	// for(pointer_type e = E-8; e < E; e++){
	//     printf("\t\t(");
	//     for(pointer_type o = 0; o < O; o++){ printf(" %d",input->entries[e].coord[o]); }
	//  	printf(")\t -> %f \n",input->entries[e].value);
	// }
	printf("\n");
}

void print_tensor_output(const outputT_t * input){
	pointer_type O = input->order;
	pointer_type E = input->entry_count;
	printf("\tOrder: %d\n",O);
	printf("\tShape:");
	for(pointer_type i = 0; i < O; i++){ printf("  %d",input->shape[i]); }
	printf("\n\tEntry Count:%d\n", E);
	for(int e = 0; e < 7; e++){
		printf("\t\t(");
		for(pointer_type o = 0; o < O; o++){ printf(" %d",input->indices[o][e]);}
		printf(")\t -> %.15f \n",input->values[e]);
	}
	printf("\t\t...\n");
	for(pointer_type e = E-8; e < E; e++){
	    printf("\t\t(");
	    for(pointer_type o = 0; o < O; o++){ printf(" %d",input->indices[o][e]); }
	 	printf(")\t -> %.15f \n",input->values[e]);
	}
	printf("\n");
}

void print_array(const pointer_type arr[], const pointer_type size){
	for(pointer_type i = 0; i < size; i++){
		printf("%d,", arr[i]);
	}
	printf("\n");
}


void print_tagged_entry(const tagged_entry ent){
	printf("%.20lf, [%lld]\n",ent.value, ent.tag);
}

T_t * tensor_alloc(pointer_type ORDER, pointer_type ENTRYCOUNT, pointer_type * SHAPE){
	T_t * new_tensor = malloc(sizeof(T_t));
	new_tensor->order = ORDER;
	new_tensor->shape = malloc(new_tensor->order * sizeof(pointer_type));

	for(int i = 0; i < new_tensor->order; ++i){
		new_tensor->shape[i] = SHAPE[i];
	}

	new_tensor->entry_count = ENTRYCOUNT; 
	new_tensor->entries = malloc(new_tensor->entry_count * sizeof(entry_t));

	for(int i = 0; i < new_tensor->entry_count; ++i){
		entry_t new_entry; 
		new_entry.coord = (pointer_type *)malloc(new_tensor->order * sizeof(pointer_type));
		new_tensor->entries[i] = new_entry; 
	}

	return new_tensor;
}

outputT_t * tensor_alloc_output(pointer_type ORDER, pointer_type ENTRYCOUNT, pointer_type * SHAPE){
	outputT_t * new_tensor = malloc(sizeof(outputT_t));
	new_tensor->order = ORDER;
	new_tensor->shape = malloc(new_tensor->order * sizeof(pointer_type));

	for(int i = 0; i < new_tensor->order; ++i){
		new_tensor->shape[i] = SHAPE[i];
	}

	new_tensor->entry_count = ENTRYCOUNT; 
	
	//ENTRIES
	new_tensor->values = (tensor_element *)malloc(new_tensor->entry_count * sizeof(tensor_element));
	new_tensor->indices = (pointer_type **)malloc(new_tensor->order * sizeof(pointer_type *));

	for(int i = 0; i < new_tensor->order; ++i){
		new_tensor->indices[i] = (pointer_type *)malloc(new_tensor->entry_count * sizeof(pointer_type));
	}

	return new_tensor;
}



T_t * tensor_read(const char * filename) {
	FILE * file_handle = fopen(filename, "r");
	if (file_handle == NULL) {
		perror("couldn't open source file");
		exit(1);
	}
	T_t * new_tensor = malloc(sizeof(T_t));
	if (new_tensor == NULL) {
		printf("failed to allcate memory\n");
		exit(1);
	}
	char buffer[4];
	char element_buffer[8];
	int read;
	//first 5 are not needed
	for(int i = 0; i < 5; i++){
		read = fread(buffer, sizeof(buffer), 1, file_handle);
	}
	
	read = fread(buffer, sizeof(buffer), 1, file_handle);
	new_tensor->order = *((int*)buffer);
	new_tensor->shape = malloc(new_tensor->order * sizeof(pointer_type));

	for(pointer_type i = 0; i < new_tensor->order; i++){
		read = fread(buffer, sizeof(buffer), 1, file_handle);
		new_tensor->shape[i] = *((int*)buffer);
	}

	read = fread(buffer, sizeof(buffer), 1, file_handle);
	new_tensor->entry_count = *((int*)buffer);
	new_tensor->entries = malloc(new_tensor->entry_count * sizeof(entry_t));

	for(pointer_type i = 0; i < new_tensor->entry_count; i++){
		entry_t new_entry;
		new_entry.coord = (pointer_type *)malloc(new_tensor->order * sizeof(pointer_type));
		new_tensor->entries[i] = new_entry;
	}

	for(pointer_type dim = 0; dim < new_tensor->order; dim++){
	for(pointer_type i = 0; i < new_tensor->entry_count; i++){
		read = fread(buffer, sizeof(buffer), 1, file_handle);
		new_tensor->entries[i].coord[dim] = *((pointer_type*)buffer); 
	}
	}

	for(pointer_type i = 0; i < new_tensor->entry_count; i++){
		read = fread(element_buffer, sizeof(element_buffer), 1, file_handle);
		new_tensor->entries[i].value = (tensor_element)*((double*)element_buffer); 
	}
	printf("read: %d\n",read);
	fclose(file_handle);
	return new_tensor;
}

void tensor_free(T_t * tensor) {
	for(int i = 0; i < tensor->entry_count; ++i){
		free(tensor->entries[i].coord);
	}
	free(tensor->entries);
	free(tensor->shape);
	free(tensor);
}

void tensor_free_output(outputT_t * tensor) {
	for(int i = 0; i < tensor->order; ++i){
		free(tensor->indices[i]);
	}
	free(tensor->indices);
	free(tensor->values); 
	free(tensor->shape);  
	free(tensor);         
}
