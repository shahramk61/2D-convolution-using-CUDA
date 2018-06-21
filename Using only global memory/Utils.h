#define BUFFER_size 256
#ifndef UTILS
#define UTILS

static inline float _min(float x, float y) {
	return x < y ? x : y;
}

static inline float _max(float x, float y) {
	return x > y ? x : y;
}

static inline float clamp(float x, float start, float end) {
	return _min(_max(x, start), end);
}

char* File_read(FILE* file, size_t size, size_t count) {
	size_t res;
	char *buffer;
	size_t bufferLen;

	if (file == NULL) {
		return NULL;
	}

	bufferLen = size * count + 1;
	buffer = (char*)malloc(sizeof(char) * bufferLen);

	res = fread(buffer, size, count, file);
	buffer[size * res] = '\0';

	return buffer;
}
static const char *skipSpaces(const char *line) {
	while (*line == ' ' || *line == '\t') {
		line++;
		if (*line == '\0') {
			break;
		}
	}
	return line;
}
static char nextNonSpaceChar(const char *line0) {
	const char *line = skipSpaces(line0);
	return *line;
}
static bool isComment(const char *line) {
	char nextChar = nextNonSpaceChar(line);
	if (nextChar == '\0') {
		return true;
	}
	else {
		return nextChar == '#';
	}
}
static char *File_readLine(FILE* file) {
	static char buffer[BUFFER_size];
	if (file == NULL) {
		return NULL;
	}
	memset(buffer, 0, BUFFER_size);

	if (fgets(buffer, BUFFER_size - 1, file)) {
		return buffer;
	}
	else {
		return NULL;
	}
}
bool File_write(FILE* file, const void *buffer, size_t size, size_t count) {
	if (file == NULL) {
		return false;
	}

	size_t res = fwrite(buffer, size, count, file);
	if (res != count) {
		printf("ERROR: Failed to write data to PPM file");
	}

	return true;
}
static char *nextLine(FILE* file) {
	char *line = NULL;
	while ((line = File_readLine(file)) != NULL) {
		if (!isComment(line)) {
			break;
		}
	}
	return line;
}

#endif //KERNELPROCESSING_UTILS_H
