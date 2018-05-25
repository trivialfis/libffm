#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <vector>
#include <map>

#include <exception>

#include <iostream>

namespace clloop
{

class CLWrapper;

// https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
static const char *get_errorstring(cl_int error)
{
  switch(error){
    // run-time and JIT compiler errors
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

static void check_code(cl_int code, std::string msg)
{
  if (code != CL_SUCCESS)
    {
      throw std::runtime_error(msg + " " + get_errorstring(code));
    }
}

struct arg_properties
{
  std::string name;

  int index;
  bool initialized;
  size_t size;
  cl_mem_flags flag;
  int arg_count;

  void *ptr;

  arg_properties ():
    index (-1),
    initialized (false),
    size (0),
    flag (CL_MEM_READ_ONLY),
    arg_count (-1),
    ptr(NULL)
  {}
  arg_properties (std::string _name,
		  int _idx,
		  bool _inited,
		  size_t _size,
		  cl_mem_flags _flag,
		  int _count) :
    name (_name),
    index (_idx),
    initialized (_inited),
    size (_size),
    flag (_flag),
    arg_count (_count),
    ptr (NULL)
  {}
};

class CLKernel
{

  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue command_queue;

  std::vector<cl_mem> mem_objects;

  std::map<std::string, arg_properties> arguments;

  cl_int cl_err;
  int arg_counter;

  size_t get_arg_index(std::string name)
  {
    auto iter = arguments.find(name);
    if (iter == arguments.end())
      {
	throw std::runtime_error("Argument: " + name + " is not defined.");
      }
    size_t index = iter->second.index;
    return index;
  }

  arg_properties create_buffer(arg_properties arg, size_t data_size)
  {
    cl_mem object;
    object = clCreateBuffer(context, arg.flag, data_size, NULL, &cl_err);
    check_code(cl_err, "Create buffer failed");
    mem_objects.push_back(object);

    arg.initialized = true;
    arg.index = mem_objects.size() - 1;
    arg.size = data_size;

    return arg;
  }

  arg_properties recreate_buffer(arg_properties arg, size_t data_size)
  {
    clReleaseMemObject(mem_objects[arg.index]);
    mem_objects[arg.index] = clCreateBuffer(context, arg.flag, data_size,
					    NULL, &cl_err);
    check_code(cl_err, "Create buffer failed");

    arg.size = data_size;

    return arg;
  }

public:

  CLKernel (cl_context _context, cl_program _program, cl_kernel _kernel,
	    cl_command_queue _cq)
    : context (_context), program (_program), kernel (_kernel),
      command_queue (_cq), cl_err (0), arg_counter (0)
  {}
  ~CLKernel ()
  {
    for (auto& obj : mem_objects)
      {
	clReleaseMemObject(obj);
      }
    clReleaseKernel(kernel);
    retrieve();
  }

  void retrieve()
  {
    for (auto& arg_pair : arguments)
      {
	cl_mem_flags flag = arg_pair.second.flag;
    	if (flag == CL_MEM_WRITE_ONLY || flag == CL_MEM_READ_WRITE)
    	  {
    	    size_t index = arg_pair.second.index;

    	    size_t size = arg_pair.second.size;
    	    void* ptr = arg_pair.second.ptr;

    	    cl_err = clEnqueueReadBuffer(command_queue, mem_objects[index],
    					 CL_FALSE, 0, size, ptr, 0, NULL, NULL);
    	    check_code(cl_err, "Reading arg: " +
    		       std::to_string(arg_pair.second.arg_count) + ": " +
		       arg_pair.second.name + " failed.");
    	  }
      }
  }

  void in(std::string name)
  {
    if (arguments.find(name) != arguments.end())
      {
	return;
      }

    arg_properties arg {name, -1, false, 0, CL_MEM_READ_ONLY, arg_counter};
    arguments[name] = arg;
    arg_counter ++;
  }
  void out(std::string name)
  {
    if (arguments.find(name) != arguments.end())
      {
	return;
      }
    arg_properties arg {name, -1, false, 0, CL_MEM_WRITE_ONLY, arg_counter};
    arguments[name] = arg;
    arg_counter ++;
  }
  void inout(std::string name)
  {
    if (arguments.find(name) != arguments.end())
      {
	return;
      }
    arg_properties arg {name, -1, false, 0, CL_MEM_READ_WRITE, arg_counter};
    arguments[name] = arg;
    arg_counter ++;
  }

  template <typename T>
  void set(std::string name, std::vector<T>& data)
  {
    size_t data_size = data.size() * sizeof(T);

    arg_properties arg = arguments[name];

    if (!arg.initialized)
      {
	arg = create_buffer(arg, data_size);
      }
    else if (arg.size != data_size)
      {
	arg = recreate_buffer(arg, data_size);
      }
    cl_err = clEnqueueWriteBuffer(command_queue, mem_objects[arg.index],
				  CL_FALSE, 0, arg.size, data.data(), 0,
				  NULL, NULL);
    check_code(cl_err, "Writing argument: " + std::to_string(arg.arg_count) + " " +
	       name + " failed. Data size: " + std::to_string(arg.size) + "\n");

    cl_err = clSetKernelArg(kernel, arg.arg_count, sizeof(cl_mem),
			    &mem_objects[arg.index]);
    check_code(cl_err, "Setting argument: " + std::to_string(arg.index) + " " +
	       name + " failed.");

    if (arg.flag == CL_MEM_WRITE_ONLY || arg.flag == CL_MEM_READ_WRITE)
      {
	arg.ptr = data.data();
      }
    arguments[name] = arg;
  }

  template <typename T>
  void set(std::string name, T* data, size_t size)
  {
    size_t data_size = size * sizeof(T);

    arg_properties arg = arguments[name];

    if (!arg.initialized)
      {
	arg = create_buffer(arg, data_size);
      }
    else if (data_size != arg.size)
      {
	arg = recreate_buffer(arg, data_size);
      }
    cl_err = clEnqueueWriteBuffer(command_queue, mem_objects[arg.index],
				  CL_FALSE, 0, data_size, (void*)data,
				  0, NULL, NULL);
    check_code(cl_err, "Writing buffer for array failed");
    cl_err = clSetKernelArg(kernel, arg.arg_count, sizeof(cl_mem),
			    &mem_objects[arg.index]);
    check_code(cl_err, "Seting argument: " + std::to_string(arg.arg_count) +
	       " " + name + " failed.");

    if (arg.flag == CL_MEM_WRITE_ONLY || arg.flag == CL_MEM_READ_WRITE)
      {
	arg.ptr = data;
      }
    arguments[name] = arg;
  }

  template <typename T>
  void set(std::string name, T scalar)
  {
    arg_properties arg = arguments[name];

    if (!arg.initialized)
      {
	arg.initialized = true;
      }
    cl_err = clSetKernelArg(kernel, arg.arg_count, sizeof(T), &scalar);
    check_code(cl_err, "Set scalar arg:" + std::to_string(arg.arg_count) +
	       " " + name + " ");
    arg.flag = CL_MEM_TYPE;
    arguments[name] = arg;
  }

  cl_context get_context() const
  {
    return context;
  }
  cl_kernel get_kernel() const
  {
    return kernel;
  }
  cl_command_queue get_command_queue() const
  {
    return command_queue;
  }

  void apply(size_t global)
  {
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, NULL, 0,
			   NULL, NULL);
    clFinish(command_queue);
  }
};

class CLWrapper
{
private:
  cl_context context;
  cl_context_properties *properties;
  cl_command_queue command_queue;
  cl_program program;
  cl_int cl_err;

  cl_uint num_of_platforms;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_uint num_of_devices;

  static constexpr size_t BUILD_LOG_SIZE = 8192;

public:
  CLWrapper()
    : num_of_platforms (0), num_of_devices (0)
  {
    cl_err = clGetPlatformIDs(1, &platform_id, &num_of_platforms);
    check_code(cl_err, "Get platform id failed.");
    cl_err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id,
			    &num_of_devices);
    check_code(cl_err, "Get device id failed.");

    properties = new cl_context_properties[3];

    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties) platform_id;
    properties[2] = 0;

    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &cl_err);
    check_code(cl_err, "Creating context failed.");

    command_queue = clCreateCommandQueue(context, device_id, 0, &cl_err);
    check_code(cl_err, "Creating command queue failed.");
  }
  ~CLWrapper()
  {
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    delete [] properties;
  }

  CLKernel eval(std::string src_path, std::string kernel_name)
  {
    FILE *fp = fopen(src_path.c_str(), "r");
    fseek(fp, 0L, SEEK_END);
    size_t filelen = ftell(fp);
    rewind(fp);

    char *kernel_src = (char*) malloc(sizeof(char) * (filelen + 1));
    size_t readlen = fread(kernel_src, 1, filelen, fp);
    if (readlen != filelen)
      {
    	fprintf(stderr, "Error reading kernel code.\n");
    	exit(1);
      }
    fclose(fp);
    kernel_src[filelen] = '\0';
    fflush(stdout);


    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src,
					NULL, &cl_err);
    if (clBuildProgram(program, 0, NULL, "-cl-opt-disable", NULL, NULL) != CL_SUCCESS)
      {
	printf("Error building program\n");
	char buffer[BUILD_LOG_SIZE];
	// get the build log
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			      sizeof(buffer), buffer, NULL);
	printf("--- Build Log -- \n %s\n", buffer);
	exit(1);
      }

    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &cl_err);

    if (cl_err != CL_SUCCESS)
      {
	fprintf(stderr, "Error creating kernel\n");
	exit(1);
      }

    return CLKernel(context, program, kernel, command_queue);
  }
};

}
