Н░.
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8├╪%
|
normalization/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namenormalization/mean
u
&normalization/mean/Read/ReadVariableOpReadVariableOpnormalization/mean*
_output_shapes
:*
dtype0
Д
normalization/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namenormalization/variance
}
*normalization/variance/Read/ReadVariableOpReadVariableOpnormalization/variance*
_output_shapes
:*
dtype0
z
normalization/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namenormalization/count
s
'normalization/count/Read/ReadVariableOpReadVariableOpnormalization/count*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
ж
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!separable_conv2d/depthwise_kernel
Я
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
:@*
dtype0
з
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!separable_conv2d/pointwise_kernel
а
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*'
_output_shapes
:@А*
dtype0
Г
separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameseparable_conv2d/bias
|
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_1/depthwise_kernel
д
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_1/pointwise_kernel
е
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_1/bias
А
+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes	
:А*
dtype0
Г
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@А*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_2/depthwise_kernel
д
7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_2/pointwise_kernel
е
7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_2/bias
А
+separable_conv2d_2/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_3/depthwise_kernel
д
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_3/pointwise_kernel
е
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_3/bias
А
+separable_conv2d_3/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_4/depthwise_kernel
д
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_4/pointwise_kernel
е
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_4/bias
А
+separable_conv2d_4/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_5/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_5/depthwise_kernel
д
7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_5/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_5/pointwise_kernel
е
7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_5/bias
А
+separable_conv2d_5/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_6/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_6/depthwise_kernel
д
7separable_conv2d_6/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_6/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_6/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_6/pointwise_kernel
е
7separable_conv2d_6/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_6/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_6/bias
А
+separable_conv2d_6/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_6/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_7/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_7/depthwise_kernel
д
7separable_conv2d_7/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_7/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_7/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_7/pointwise_kernel
е
7separable_conv2d_7/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_7/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_7/bias
А
+separable_conv2d_7/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_7/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_8/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_8/depthwise_kernel
д
7separable_conv2d_8/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_8/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_8/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_8/pointwise_kernel
е
7separable_conv2d_8/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_8/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_8/bias
А
+separable_conv2d_8/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_8/bias*
_output_shapes	
:А*
dtype0
л
#separable_conv2d_9/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_9/depthwise_kernel
д
7separable_conv2d_9/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_9/depthwise_kernel*'
_output_shapes
:А*
dtype0
м
#separable_conv2d_9/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_9/pointwise_kernel
е
7separable_conv2d_9/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_9/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_9/bias
А
+separable_conv2d_9/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_9/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_10/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_10/depthwise_kernel
ж
8separable_conv2d_10/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_10/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_10/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_10/pointwise_kernel
з
8separable_conv2d_10/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_10/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_10/bias
В
,separable_conv2d_10/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_10/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_11/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_11/depthwise_kernel
ж
8separable_conv2d_11/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_11/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_11/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_11/pointwise_kernel
з
8separable_conv2d_11/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_11/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_11/bias
В
,separable_conv2d_11/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_11/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_12/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_12/depthwise_kernel
ж
8separable_conv2d_12/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_12/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_12/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_12/pointwise_kernel
з
8separable_conv2d_12/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_12/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_12/bias
В
,separable_conv2d_12/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_12/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_13/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_13/depthwise_kernel
ж
8separable_conv2d_13/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_13/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_13/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_13/pointwise_kernel
з
8separable_conv2d_13/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_13/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_13/bias
В
,separable_conv2d_13/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_13/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_14/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_14/depthwise_kernel
ж
8separable_conv2d_14/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_14/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_14/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_14/pointwise_kernel
з
8separable_conv2d_14/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_14/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_14/bias
В
,separable_conv2d_14/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_14/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_15/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_15/depthwise_kernel
ж
8separable_conv2d_15/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_15/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_15/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_15/pointwise_kernel
з
8separable_conv2d_15/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_15/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_15/bias
В
,separable_conv2d_15/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_15/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:А*
dtype0
Н
regression_head_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameregression_head_1/kernel
Ж
,regression_head_1/kernel/Read/ReadVariableOpReadVariableOpregression_head_1/kernel*
_output_shapes
:	А*
dtype0
Д
regression_head_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameregression_head_1/bias
}
*regression_head_1/bias/Read/ReadVariableOpReadVariableOpregression_head_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
В
conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d/kernel/m
{
#conv2d/kernel/m/Read/ReadVariableOpReadVariableOpconv2d/kernel/m*&
_output_shapes
:@*
dtype0
r
conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias/m
k
!conv2d/bias/m/Read/ReadVariableOpReadVariableOpconv2d/bias/m*
_output_shapes
:@*
dtype0
к
#separable_conv2d/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d/depthwise_kernel/m
г
7separable_conv2d/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp#separable_conv2d/depthwise_kernel/m*&
_output_shapes
:@*
dtype0
л
#separable_conv2d/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*4
shared_name%#separable_conv2d/pointwise_kernel/m
д
7separable_conv2d/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp#separable_conv2d/pointwise_kernel/m*'
_output_shapes
:@А*
dtype0
З
separable_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d/bias/m
А
+separable_conv2d/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_1/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_1/depthwise_kernel/m
и
9separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_1/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_1/pointwise_kernel/m
й
9separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_1/bias/m
Д
-separable_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias/m*
_output_shapes	
:А*
dtype0
З
conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_1/kernel/m
А
%conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/m*'
_output_shapes
:@А*
dtype0
w
conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_1/bias/m
p
#conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpconv2d_1/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_2/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_2/depthwise_kernel/m
и
9separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_2/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_2/pointwise_kernel/m
й
9separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_2/bias/m
Д
-separable_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_3/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_3/depthwise_kernel/m
и
9separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_3/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_3/pointwise_kernel/m
й
9separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_3/bias/m
Д
-separable_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_4/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_4/depthwise_kernel/m
и
9separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_4/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_4/pointwise_kernel/m
й
9separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_4/bias/m
Д
-separable_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_5/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_5/depthwise_kernel/m
и
9separable_conv2d_5/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_5/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_5/pointwise_kernel/m
й
9separable_conv2d_5/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_5/bias/m
Д
-separable_conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_6/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_6/depthwise_kernel/m
и
9separable_conv2d_6/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_6/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_6/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_6/pointwise_kernel/m
й
9separable_conv2d_6/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_6/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_6/bias/m
Д
-separable_conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_6/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_7/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_7/depthwise_kernel/m
и
9separable_conv2d_7/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_7/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_7/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_7/pointwise_kernel/m
й
9separable_conv2d_7/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_7/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_7/bias/m
Д
-separable_conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_7/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_8/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_8/depthwise_kernel/m
и
9separable_conv2d_8/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_8/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_8/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_8/pointwise_kernel/m
й
9separable_conv2d_8/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_8/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_8/bias/m
Д
-separable_conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_8/bias/m*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_9/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_9/depthwise_kernel/m
и
9separable_conv2d_9/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_9/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_9/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_9/pointwise_kernel/m
й
9separable_conv2d_9/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_9/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_9/bias/m
Д
-separable_conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_9/bias/m*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_10/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_10/depthwise_kernel/m
к
:separable_conv2d_10/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_10/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_10/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_10/pointwise_kernel/m
л
:separable_conv2d_10/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_10/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_10/bias/m
Ж
.separable_conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_10/bias/m*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_11/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_11/depthwise_kernel/m
к
:separable_conv2d_11/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_11/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_11/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_11/pointwise_kernel/m
л
:separable_conv2d_11/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_11/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_11/bias/m
Ж
.separable_conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_11/bias/m*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_12/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_12/depthwise_kernel/m
к
:separable_conv2d_12/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_12/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_12/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_12/pointwise_kernel/m
л
:separable_conv2d_12/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_12/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_12/bias/m
Ж
.separable_conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_12/bias/m*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_13/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_13/depthwise_kernel/m
к
:separable_conv2d_13/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_13/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_13/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_13/pointwise_kernel/m
л
:separable_conv2d_13/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_13/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_13/bias/m
Ж
.separable_conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_13/bias/m*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_14/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_14/depthwise_kernel/m
к
:separable_conv2d_14/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_14/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_14/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_14/pointwise_kernel/m
л
:separable_conv2d_14/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_14/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_14/bias/m
Ж
.separable_conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_14/bias/m*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_15/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_15/depthwise_kernel/m
к
:separable_conv2d_15/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_15/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_15/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_15/pointwise_kernel/m
л
:separable_conv2d_15/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp&separable_conv2d_15/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_15/bias/m
Ж
.separable_conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_15/bias/m*
_output_shapes	
:А*
dtype0
И
conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameconv2d_2/kernel/m
Б
%conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/m*(
_output_shapes
:АА*
dtype0
w
conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_2/bias/m
p
#conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpconv2d_2/bias/m*
_output_shapes	
:А*
dtype0
С
regression_head_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*+
shared_nameregression_head_1/kernel/m
К
.regression_head_1/kernel/m/Read/ReadVariableOpReadVariableOpregression_head_1/kernel/m*
_output_shapes
:	А*
dtype0
И
regression_head_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameregression_head_1/bias/m
Б
,regression_head_1/bias/m/Read/ReadVariableOpReadVariableOpregression_head_1/bias/m*
_output_shapes
:*
dtype0
В
conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d/kernel/v
{
#conv2d/kernel/v/Read/ReadVariableOpReadVariableOpconv2d/kernel/v*&
_output_shapes
:@*
dtype0
r
conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias/v
k
!conv2d/bias/v/Read/ReadVariableOpReadVariableOpconv2d/bias/v*
_output_shapes
:@*
dtype0
к
#separable_conv2d/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d/depthwise_kernel/v
г
7separable_conv2d/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp#separable_conv2d/depthwise_kernel/v*&
_output_shapes
:@*
dtype0
л
#separable_conv2d/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*4
shared_name%#separable_conv2d/pointwise_kernel/v
д
7separable_conv2d/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp#separable_conv2d/pointwise_kernel/v*'
_output_shapes
:@А*
dtype0
З
separable_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d/bias/v
А
+separable_conv2d/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_1/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_1/depthwise_kernel/v
и
9separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_1/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_1/pointwise_kernel/v
й
9separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_1/bias/v
Д
-separable_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias/v*
_output_shapes	
:А*
dtype0
З
conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_1/kernel/v
А
%conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/v*'
_output_shapes
:@А*
dtype0
w
conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_1/bias/v
p
#conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpconv2d_1/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_2/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_2/depthwise_kernel/v
и
9separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_2/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_2/pointwise_kernel/v
й
9separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_2/bias/v
Д
-separable_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_3/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_3/depthwise_kernel/v
и
9separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_3/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_3/pointwise_kernel/v
й
9separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_3/bias/v
Д
-separable_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_4/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_4/depthwise_kernel/v
и
9separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_4/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_4/pointwise_kernel/v
й
9separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_4/bias/v
Д
-separable_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_5/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_5/depthwise_kernel/v
и
9separable_conv2d_5/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_5/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_5/pointwise_kernel/v
й
9separable_conv2d_5/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_5/bias/v
Д
-separable_conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_6/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_6/depthwise_kernel/v
и
9separable_conv2d_6/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_6/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_6/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_6/pointwise_kernel/v
й
9separable_conv2d_6/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_6/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_6/bias/v
Д
-separable_conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_6/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_7/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_7/depthwise_kernel/v
и
9separable_conv2d_7/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_7/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_7/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_7/pointwise_kernel/v
й
9separable_conv2d_7/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_7/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_7/bias/v
Д
-separable_conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_7/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_8/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_8/depthwise_kernel/v
и
9separable_conv2d_8/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_8/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_8/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_8/pointwise_kernel/v
й
9separable_conv2d_8/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_8/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_8/bias/v
Д
-separable_conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_8/bias/v*
_output_shapes	
:А*
dtype0
п
%separable_conv2d_9/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_9/depthwise_kernel/v
и
9separable_conv2d_9/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_9/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
░
%separable_conv2d_9/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_9/pointwise_kernel/v
й
9separable_conv2d_9/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_9/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_9/bias/v
Д
-separable_conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_9/bias/v*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_10/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_10/depthwise_kernel/v
к
:separable_conv2d_10/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_10/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_10/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_10/pointwise_kernel/v
л
:separable_conv2d_10/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_10/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_10/bias/v
Ж
.separable_conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_10/bias/v*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_11/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_11/depthwise_kernel/v
к
:separable_conv2d_11/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_11/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_11/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_11/pointwise_kernel/v
л
:separable_conv2d_11/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_11/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_11/bias/v
Ж
.separable_conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_11/bias/v*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_12/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_12/depthwise_kernel/v
к
:separable_conv2d_12/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_12/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_12/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_12/pointwise_kernel/v
л
:separable_conv2d_12/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_12/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_12/bias/v
Ж
.separable_conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_12/bias/v*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_13/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_13/depthwise_kernel/v
к
:separable_conv2d_13/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_13/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_13/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_13/pointwise_kernel/v
л
:separable_conv2d_13/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_13/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_13/bias/v
Ж
.separable_conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_13/bias/v*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_14/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_14/depthwise_kernel/v
к
:separable_conv2d_14/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_14/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_14/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_14/pointwise_kernel/v
л
:separable_conv2d_14/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_14/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_14/bias/v
Ж
.separable_conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_14/bias/v*
_output_shapes	
:А*
dtype0
▒
&separable_conv2d_15/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&separable_conv2d_15/depthwise_kernel/v
к
:separable_conv2d_15/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_15/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
▓
&separable_conv2d_15/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*7
shared_name(&separable_conv2d_15/pointwise_kernel/v
л
:separable_conv2d_15/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp&separable_conv2d_15/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Н
separable_conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameseparable_conv2d_15/bias/v
Ж
.separable_conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_15/bias/v*
_output_shapes	
:А*
dtype0
И
conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameconv2d_2/kernel/v
Б
%conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/v*(
_output_shapes
:АА*
dtype0
w
conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_2/bias/v
p
#conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpconv2d_2/bias/v*
_output_shapes	
:А*
dtype0
С
regression_head_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*+
shared_nameregression_head_1/kernel/v
К
.regression_head_1/kernel/v/Read/ReadVariableOpReadVariableOpregression_head_1/kernel/v*
_output_shapes
:	А*
dtype0
И
regression_head_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameregression_head_1/bias/v
Б
,regression_head_1/bias/v/Read/ReadVariableOpReadVariableOpregression_head_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ив
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*тб
value╫бB╙б B╦б
╘
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer-21
layer_with_weights-15
layer-22
layer_with_weights-16
layer-23
layer-24
layer_with_weights-17
layer-25
layer_with_weights-18
layer-26
layer-27
layer_with_weights-19
layer-28
layer-29
layer-30
 layer_with_weights-20
 layer-31
!	optimizer
"regularization_losses
#	variables
$trainable_variables
%	keras_api
&
signatures
 
а
'state_variables
(_broadcast_shape
)mean
*variance
	+count
,regularization_losses
-	variables
.trainable_variables
/	keras_api
h

0kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
И
6depthwise_kernel
7pointwise_kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
И
=depthwise_kernel
>pointwise_kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
И
Ndepthwise_kernel
Opointwise_kernel
Pbias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
И
Udepthwise_kernel
Vpointwise_kernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
И
`depthwise_kernel
apointwise_kernel
bbias
cregularization_losses
d	variables
etrainable_variables
f	keras_api
И
gdepthwise_kernel
hpointwise_kernel
ibias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
R
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
И
rdepthwise_kernel
spointwise_kernel
tbias
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
И
ydepthwise_kernel
zpointwise_kernel
{bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
V
Аregularization_losses
Б	variables
Вtrainable_variables
Г	keras_api
П
Дdepthwise_kernel
Еpointwise_kernel
	Жbias
Зregularization_losses
И	variables
Йtrainable_variables
К	keras_api
П
Лdepthwise_kernel
Мpointwise_kernel
	Нbias
Оregularization_losses
П	variables
Рtrainable_variables
С	keras_api
V
Тregularization_losses
У	variables
Фtrainable_variables
Х	keras_api
П
Цdepthwise_kernel
Чpointwise_kernel
	Шbias
Щregularization_losses
Ъ	variables
Ыtrainable_variables
Ь	keras_api
П
Эdepthwise_kernel
Юpointwise_kernel
	Яbias
аregularization_losses
б	variables
вtrainable_variables
г	keras_api
V
дregularization_losses
е	variables
жtrainable_variables
з	keras_api
П
иdepthwise_kernel
йpointwise_kernel
	кbias
лregularization_losses
м	variables
нtrainable_variables
о	keras_api
П
пdepthwise_kernel
░pointwise_kernel
	▒bias
▓regularization_losses
│	variables
┤trainable_variables
╡	keras_api
V
╢regularization_losses
╖	variables
╕trainable_variables
╣	keras_api
П
║depthwise_kernel
╗pointwise_kernel
	╝bias
╜regularization_losses
╛	variables
┐trainable_variables
└	keras_api
П
┴depthwise_kernel
┬pointwise_kernel
	├bias
─regularization_losses
┼	variables
╞trainable_variables
╟	keras_api
V
╚regularization_losses
╔	variables
╩trainable_variables
╦	keras_api
n
╠kernel
	═bias
╬regularization_losses
╧	variables
╨trainable_variables
╤	keras_api
V
╥regularization_losses
╙	variables
╘trainable_variables
╒	keras_api
V
╓regularization_losses
╫	variables
╪trainable_variables
┘	keras_api
n
┌kernel
	█bias
▄regularization_losses
▌	variables
▐trainable_variables
▀	keras_api
Ш	0mь1mэ6mю7mя8mЁ=mё>mЄ?mєDmЇEmїNmЎOmўPm°Um∙Vm·Wm√`m№am¤bm■gm hmАimБrmВsmГtmДymЕzmЖ{mЗ	ДmИ	ЕmЙ	ЖmК	ЛmЛ	МmМ	НmН	ЦmО	ЧmП	ШmР	ЭmС	ЮmТ	ЯmУ	иmФ	йmХ	кmЦ	пmЧ	░mШ	▒mЩ	║mЪ	╗mЫ	╝mЬ	┴mЭ	┬mЮ	├mЯ	╠mа	═mб	┌mв	█mг0vд1vе6vж7vз8vи=vй>vк?vлDvмEvнNvоOvпPv░Uv▒Vv▓Wv│`v┤av╡bv╢gv╖hv╕iv╣rv║sv╗tv╝yv╜zv╛{v┐	Дv└	Еv┴	Жv┬	Лv├	Мv─	Нv┼	Цv╞	Чv╟	Шv╚	Эv╔	Юv╩	Яv╦	иv╠	йv═	кv╬	пv╧	░v╨	▒v╤	║v╥	╗v╙	╝v╘	┴v╒	┬v╓	├v╫	╠v╪	═v┘	┌v┌	█v█
 
ъ
)0
*1
+2
03
14
65
76
87
=8
>9
?10
D11
E12
N13
O14
P15
U16
V17
W18
`19
a20
b21
g22
h23
i24
r25
s26
t27
y28
z29
{30
Д31
Е32
Ж33
Л34
М35
Н36
Ц37
Ч38
Ш39
Э40
Ю41
Я42
и43
й44
к45
п46
░47
▒48
║49
╗50
╝51
┴52
┬53
├54
╠55
═56
┌57
█58
╥
00
11
62
73
84
=5
>6
?7
D8
E9
N10
O11
P12
U13
V14
W15
`16
a17
b18
g19
h20
i21
r22
s23
t24
y25
z26
{27
Д28
Е29
Ж30
Л31
М32
Н33
Ц34
Ч35
Ш36
Э37
Ю38
Я39
и40
й41
к42
п43
░44
▒45
║46
╗47
╝48
┴49
┬50
├51
╠52
═53
┌54
█55
Ю
"regularization_losses
 рlayer_regularization_losses
сlayers
тnon_trainable_variables
#	variables
$trainable_variables
уmetrics
 
#
)mean
*variance
	+count
 
\Z
VARIABLE_VALUEnormalization/mean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEnormalization/variance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEnormalization/count5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1
+2
 
Ю
,regularization_losses
 фlayer_regularization_losses
хlayers
цnon_trainable_variables
-	variables
.trainable_variables
чmetrics
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
Ю
2regularization_losses
 шlayer_regularization_losses
щlayers
ъnon_trainable_variables
3	variables
4trainable_variables
ыmetrics
wu
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71
82

60
71
82
Ю
9regularization_losses
 ьlayer_regularization_losses
эlayers
юnon_trainable_variables
:	variables
;trainable_variables
яmetrics
yw
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1
?2

=0
>1
?2
Ю
@regularization_losses
 Ёlayer_regularization_losses
ёlayers
Єnon_trainable_variables
A	variables
Btrainable_variables
єmetrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
Ю
Fregularization_losses
 Їlayer_regularization_losses
їlayers
Ўnon_trainable_variables
G	variables
Htrainable_variables
ўmetrics
 
 
 
Ю
Jregularization_losses
 °layer_regularization_losses
∙layers
·non_trainable_variables
K	variables
Ltrainable_variables
√metrics
yw
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1
P2

N0
O1
P2
Ю
Qregularization_losses
 №layer_regularization_losses
¤layers
■non_trainable_variables
R	variables
Strainable_variables
 metrics
yw
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1
W2

U0
V1
W2
Ю
Xregularization_losses
 Аlayer_regularization_losses
Бlayers
Вnon_trainable_variables
Y	variables
Ztrainable_variables
Гmetrics
 
 
 
Ю
\regularization_losses
 Дlayer_regularization_losses
Еlayers
Жnon_trainable_variables
]	variables
^trainable_variables
Зmetrics
yw
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1
b2

`0
a1
b2
Ю
cregularization_losses
 Иlayer_regularization_losses
Йlayers
Кnon_trainable_variables
d	variables
etrainable_variables
Лmetrics
yw
VARIABLE_VALUE#separable_conv2d_5/depthwise_kernel@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_5/pointwise_kernel@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1
i2

g0
h1
i2
Ю
jregularization_losses
 Мlayer_regularization_losses
Нlayers
Оnon_trainable_variables
k	variables
ltrainable_variables
Пmetrics
 
 
 
Ю
nregularization_losses
 Рlayer_regularization_losses
Сlayers
Тnon_trainable_variables
o	variables
ptrainable_variables
Уmetrics
yw
VARIABLE_VALUE#separable_conv2d_6/depthwise_kernel@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_6/pointwise_kernel@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_6/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1
t2

r0
s1
t2
Ю
uregularization_losses
 Фlayer_regularization_losses
Хlayers
Цnon_trainable_variables
v	variables
wtrainable_variables
Чmetrics
zx
VARIABLE_VALUE#separable_conv2d_7/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#separable_conv2d_7/pointwise_kernelAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_7/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

y0
z1
{2

y0
z1
{2
Ю
|regularization_losses
 Шlayer_regularization_losses
Щlayers
Ъnon_trainable_variables
}	variables
~trainable_variables
Ыmetrics
 
 
 
б
Аregularization_losses
 Ьlayer_regularization_losses
Эlayers
Юnon_trainable_variables
Б	variables
Вtrainable_variables
Яmetrics
zx
VARIABLE_VALUE#separable_conv2d_8/depthwise_kernelAlayer_with_weights-11/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#separable_conv2d_8/pointwise_kernelAlayer_with_weights-11/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_8/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Д0
Е1
Ж2

Д0
Е1
Ж2
б
Зregularization_losses
 аlayer_regularization_losses
бlayers
вnon_trainable_variables
И	variables
Йtrainable_variables
гmetrics
zx
VARIABLE_VALUE#separable_conv2d_9/depthwise_kernelAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#separable_conv2d_9/pointwise_kernelAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_9/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Л0
М1
Н2

Л0
М1
Н2
б
Оregularization_losses
 дlayer_regularization_losses
еlayers
жnon_trainable_variables
П	variables
Рtrainable_variables
зmetrics
 
 
 
б
Тregularization_losses
 иlayer_regularization_losses
йlayers
кnon_trainable_variables
У	variables
Фtrainable_variables
лmetrics
{y
VARIABLE_VALUE$separable_conv2d_10/depthwise_kernelAlayer_with_weights-13/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_10/pointwise_kernelAlayer_with_weights-13/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_10/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ц0
Ч1
Ш2

Ц0
Ч1
Ш2
б
Щregularization_losses
 мlayer_regularization_losses
нlayers
оnon_trainable_variables
Ъ	variables
Ыtrainable_variables
пmetrics
{y
VARIABLE_VALUE$separable_conv2d_11/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_11/pointwise_kernelAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_11/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Э0
Ю1
Я2

Э0
Ю1
Я2
б
аregularization_losses
 ░layer_regularization_losses
▒layers
▓non_trainable_variables
б	variables
вtrainable_variables
│metrics
 
 
 
б
дregularization_losses
 ┤layer_regularization_losses
╡layers
╢non_trainable_variables
е	variables
жtrainable_variables
╖metrics
{y
VARIABLE_VALUE$separable_conv2d_12/depthwise_kernelAlayer_with_weights-15/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_12/pointwise_kernelAlayer_with_weights-15/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_12/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

и0
й1
к2

и0
й1
к2
б
лregularization_losses
 ╕layer_regularization_losses
╣layers
║non_trainable_variables
м	variables
нtrainable_variables
╗metrics
{y
VARIABLE_VALUE$separable_conv2d_13/depthwise_kernelAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_13/pointwise_kernelAlayer_with_weights-16/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_13/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

п0
░1
▒2

п0
░1
▒2
б
▓regularization_losses
 ╝layer_regularization_losses
╜layers
╛non_trainable_variables
│	variables
┤trainable_variables
┐metrics
 
 
 
б
╢regularization_losses
 └layer_regularization_losses
┴layers
┬non_trainable_variables
╖	variables
╕trainable_variables
├metrics
{y
VARIABLE_VALUE$separable_conv2d_14/depthwise_kernelAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_14/pointwise_kernelAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_14/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

║0
╗1
╝2

║0
╗1
╝2
б
╜regularization_losses
 ─layer_regularization_losses
┼layers
╞non_trainable_variables
╛	variables
┐trainable_variables
╟metrics
{y
VARIABLE_VALUE$separable_conv2d_15/depthwise_kernelAlayer_with_weights-18/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_15/pointwise_kernelAlayer_with_weights-18/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_15/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

┴0
┬1
├2

┴0
┬1
├2
б
─regularization_losses
 ╚layer_regularization_losses
╔layers
╩non_trainable_variables
┼	variables
╞trainable_variables
╦metrics
 
 
 
б
╚regularization_losses
 ╠layer_regularization_losses
═layers
╬non_trainable_variables
╔	variables
╩trainable_variables
╧metrics
\Z
VARIABLE_VALUEconv2d_2/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_2/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

╠0
═1

╠0
═1
б
╬regularization_losses
 ╨layer_regularization_losses
╤layers
╥non_trainable_variables
╧	variables
╨trainable_variables
╙metrics
 
 
 
б
╥regularization_losses
 ╘layer_regularization_losses
╒layers
╓non_trainable_variables
╙	variables
╘trainable_variables
╫metrics
 
 
 
б
╓regularization_losses
 ╪layer_regularization_losses
┘layers
┌non_trainable_variables
╫	variables
╪trainable_variables
█metrics
ec
VARIABLE_VALUEregression_head_1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEregression_head_1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

┌0
█1

┌0
█1
б
▄regularization_losses
 ▄layer_regularization_losses
▌layers
▐non_trainable_variables
▌	variables
▐trainable_variables
▀metrics
 
Ў
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31

)0
*1
+2

р0
 
 

)0
*1
+2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


сtotal

тcount
у
_fn_kwargs
фregularization_losses
х	variables
цtrainable_variables
ч	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

с0
т1
 
б
фregularization_losses
 шlayer_regularization_losses
щlayers
ъnon_trainable_variables
х	variables
цtrainable_variables
ыmetrics
 
 

с0
т1
 
wu
VARIABLE_VALUEconv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEseparable_conv2d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/depthwise_kernel/m\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/pointwise_kernel/m\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/depthwise_kernel/m\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/pointwise_kernel/m\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/depthwise_kernel/m\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/pointwise_kernel/m\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/depthwise_kernel/m\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/pointwise_kernel/m\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/depthwise_kernel/m\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/pointwise_kernel/m\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_6/depthwise_kernel/m\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_6/pointwise_kernel/m\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_6/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_7/depthwise_kernel/m]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_7/pointwise_kernel/m]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEseparable_conv2d_7/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_8/depthwise_kernel/m]layer_with_weights-11/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_8/pointwise_kernel/m]layer_with_weights-11/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEseparable_conv2d_8/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_9/depthwise_kernel/m]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_9/pointwise_kernel/m]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEseparable_conv2d_9/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_10/depthwise_kernel/m]layer_with_weights-13/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_10/pointwise_kernel/m]layer_with_weights-13/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_10/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_11/depthwise_kernel/m]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_11/pointwise_kernel/m]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_11/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_12/depthwise_kernel/m]layer_with_weights-15/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_12/pointwise_kernel/m]layer_with_weights-15/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_12/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_13/depthwise_kernel/m]layer_with_weights-16/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_13/pointwise_kernel/m]layer_with_weights-16/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_13/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_14/depthwise_kernel/m]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_14/pointwise_kernel/m]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_14/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_15/depthwise_kernel/m]layer_with_weights-18/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_15/pointwise_kernel/m]layer_with_weights-18/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_15/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_2/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_2/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEregression_head_1/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEregression_head_1/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEconv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEseparable_conv2d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/depthwise_kernel/v\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/pointwise_kernel/v\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/depthwise_kernel/v\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/pointwise_kernel/v\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/depthwise_kernel/v\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/pointwise_kernel/v\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/depthwise_kernel/v\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/pointwise_kernel/v\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/depthwise_kernel/v\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/pointwise_kernel/v\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_6/depthwise_kernel/v\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_6/pointwise_kernel/v\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_6/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_7/depthwise_kernel/v]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_7/pointwise_kernel/v]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEseparable_conv2d_7/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_8/depthwise_kernel/v]layer_with_weights-11/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_8/pointwise_kernel/v]layer_with_weights-11/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEseparable_conv2d_8/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_9/depthwise_kernel/v]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЩЦ
VARIABLE_VALUE%separable_conv2d_9/pointwise_kernel/v]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEseparable_conv2d_9/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_10/depthwise_kernel/v]layer_with_weights-13/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_10/pointwise_kernel/v]layer_with_weights-13/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_10/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_11/depthwise_kernel/v]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_11/pointwise_kernel/v]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_11/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_12/depthwise_kernel/v]layer_with_weights-15/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_12/pointwise_kernel/v]layer_with_weights-15/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_12/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_13/depthwise_kernel/v]layer_with_weights-16/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_13/pointwise_kernel/v]layer_with_weights-16/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_13/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_14/depthwise_kernel/v]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_14/pointwise_kernel/v]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_14/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_15/depthwise_kernel/v]layer_with_weights-18/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE&separable_conv2d_15/pointwise_kernel/v]layer_with_weights-18/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEseparable_conv2d_15/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_2/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_2/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEregression_head_1/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEregression_head_1/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         @@*
dtype0*$
shape:         @@
╠
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1normalization/meannormalization/varianceconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/bias#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/bias#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelseparable_conv2d_6/bias#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelseparable_conv2d_7/bias#separable_conv2d_8/depthwise_kernel#separable_conv2d_8/pointwise_kernelseparable_conv2d_8/bias#separable_conv2d_9/depthwise_kernel#separable_conv2d_9/pointwise_kernelseparable_conv2d_9/bias$separable_conv2d_10/depthwise_kernel$separable_conv2d_10/pointwise_kernelseparable_conv2d_10/bias$separable_conv2d_11/depthwise_kernel$separable_conv2d_11/pointwise_kernelseparable_conv2d_11/bias$separable_conv2d_12/depthwise_kernel$separable_conv2d_12/pointwise_kernelseparable_conv2d_12/bias$separable_conv2d_13/depthwise_kernel$separable_conv2d_13/pointwise_kernelseparable_conv2d_13/bias$separable_conv2d_14/depthwise_kernel$separable_conv2d_14/pointwise_kernelseparable_conv2d_14/bias$separable_conv2d_15/depthwise_kernel$separable_conv2d_15/pointwise_kernelseparable_conv2d_15/biasconv2d_2/kernelconv2d_2/biasregression_head_1/kernelregression_head_1/bias*F
Tin?
=2;*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_80753
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╔I
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&normalization/mean/Read/ReadVariableOp*normalization/variance/Read/ReadVariableOp'normalization/count/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_2/bias/Read/ReadVariableOp7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_5/bias/Read/ReadVariableOp7separable_conv2d_6/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_6/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_6/bias/Read/ReadVariableOp7separable_conv2d_7/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_7/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_7/bias/Read/ReadVariableOp7separable_conv2d_8/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_8/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_8/bias/Read/ReadVariableOp7separable_conv2d_9/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_9/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_9/bias/Read/ReadVariableOp8separable_conv2d_10/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_10/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_10/bias/Read/ReadVariableOp8separable_conv2d_11/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_11/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_11/bias/Read/ReadVariableOp8separable_conv2d_12/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_12/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_12/bias/Read/ReadVariableOp8separable_conv2d_13/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_13/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_13/bias/Read/ReadVariableOp8separable_conv2d_14/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_14/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_14/bias/Read/ReadVariableOp8separable_conv2d_15/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_15/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_15/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp,regression_head_1/kernel/Read/ReadVariableOp*regression_head_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#conv2d/kernel/m/Read/ReadVariableOp!conv2d/bias/m/Read/ReadVariableOp7separable_conv2d/depthwise_kernel/m/Read/ReadVariableOp7separable_conv2d/pointwise_kernel/m/Read/ReadVariableOp+separable_conv2d/bias/m/Read/ReadVariableOp9separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_1/bias/m/Read/ReadVariableOp%conv2d_1/kernel/m/Read/ReadVariableOp#conv2d_1/bias/m/Read/ReadVariableOp9separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_2/bias/m/Read/ReadVariableOp9separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_3/bias/m/Read/ReadVariableOp9separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_4/bias/m/Read/ReadVariableOp9separable_conv2d_5/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_5/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_5/bias/m/Read/ReadVariableOp9separable_conv2d_6/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_6/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_6/bias/m/Read/ReadVariableOp9separable_conv2d_7/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_7/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_7/bias/m/Read/ReadVariableOp9separable_conv2d_8/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_8/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_8/bias/m/Read/ReadVariableOp9separable_conv2d_9/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_9/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_9/bias/m/Read/ReadVariableOp:separable_conv2d_10/depthwise_kernel/m/Read/ReadVariableOp:separable_conv2d_10/pointwise_kernel/m/Read/ReadVariableOp.separable_conv2d_10/bias/m/Read/ReadVariableOp:separable_conv2d_11/depthwise_kernel/m/Read/ReadVariableOp:separable_conv2d_11/pointwise_kernel/m/Read/ReadVariableOp.separable_conv2d_11/bias/m/Read/ReadVariableOp:separable_conv2d_12/depthwise_kernel/m/Read/ReadVariableOp:separable_conv2d_12/pointwise_kernel/m/Read/ReadVariableOp.separable_conv2d_12/bias/m/Read/ReadVariableOp:separable_conv2d_13/depthwise_kernel/m/Read/ReadVariableOp:separable_conv2d_13/pointwise_kernel/m/Read/ReadVariableOp.separable_conv2d_13/bias/m/Read/ReadVariableOp:separable_conv2d_14/depthwise_kernel/m/Read/ReadVariableOp:separable_conv2d_14/pointwise_kernel/m/Read/ReadVariableOp.separable_conv2d_14/bias/m/Read/ReadVariableOp:separable_conv2d_15/depthwise_kernel/m/Read/ReadVariableOp:separable_conv2d_15/pointwise_kernel/m/Read/ReadVariableOp.separable_conv2d_15/bias/m/Read/ReadVariableOp%conv2d_2/kernel/m/Read/ReadVariableOp#conv2d_2/bias/m/Read/ReadVariableOp.regression_head_1/kernel/m/Read/ReadVariableOp,regression_head_1/bias/m/Read/ReadVariableOp#conv2d/kernel/v/Read/ReadVariableOp!conv2d/bias/v/Read/ReadVariableOp7separable_conv2d/depthwise_kernel/v/Read/ReadVariableOp7separable_conv2d/pointwise_kernel/v/Read/ReadVariableOp+separable_conv2d/bias/v/Read/ReadVariableOp9separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_1/bias/v/Read/ReadVariableOp%conv2d_1/kernel/v/Read/ReadVariableOp#conv2d_1/bias/v/Read/ReadVariableOp9separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_2/bias/v/Read/ReadVariableOp9separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_3/bias/v/Read/ReadVariableOp9separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_4/bias/v/Read/ReadVariableOp9separable_conv2d_5/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_5/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_5/bias/v/Read/ReadVariableOp9separable_conv2d_6/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_6/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_6/bias/v/Read/ReadVariableOp9separable_conv2d_7/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_7/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_7/bias/v/Read/ReadVariableOp9separable_conv2d_8/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_8/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_8/bias/v/Read/ReadVariableOp9separable_conv2d_9/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_9/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_9/bias/v/Read/ReadVariableOp:separable_conv2d_10/depthwise_kernel/v/Read/ReadVariableOp:separable_conv2d_10/pointwise_kernel/v/Read/ReadVariableOp.separable_conv2d_10/bias/v/Read/ReadVariableOp:separable_conv2d_11/depthwise_kernel/v/Read/ReadVariableOp:separable_conv2d_11/pointwise_kernel/v/Read/ReadVariableOp.separable_conv2d_11/bias/v/Read/ReadVariableOp:separable_conv2d_12/depthwise_kernel/v/Read/ReadVariableOp:separable_conv2d_12/pointwise_kernel/v/Read/ReadVariableOp.separable_conv2d_12/bias/v/Read/ReadVariableOp:separable_conv2d_13/depthwise_kernel/v/Read/ReadVariableOp:separable_conv2d_13/pointwise_kernel/v/Read/ReadVariableOp.separable_conv2d_13/bias/v/Read/ReadVariableOp:separable_conv2d_14/depthwise_kernel/v/Read/ReadVariableOp:separable_conv2d_14/pointwise_kernel/v/Read/ReadVariableOp.separable_conv2d_14/bias/v/Read/ReadVariableOp:separable_conv2d_15/depthwise_kernel/v/Read/ReadVariableOp:separable_conv2d_15/pointwise_kernel/v/Read/ReadVariableOp.separable_conv2d_15/bias/v/Read/ReadVariableOp%conv2d_2/kernel/v/Read/ReadVariableOp#conv2d_2/bias/v/Read/ReadVariableOp.regression_head_1/kernel/v/Read/ReadVariableOp,regression_head_1/bias/v/Read/ReadVariableOpConst*╜
Tin╡
▓2п*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_82043
└.
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenormalization/meannormalization/variancenormalization/countconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/bias#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/bias#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelseparable_conv2d_6/bias#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelseparable_conv2d_7/bias#separable_conv2d_8/depthwise_kernel#separable_conv2d_8/pointwise_kernelseparable_conv2d_8/bias#separable_conv2d_9/depthwise_kernel#separable_conv2d_9/pointwise_kernelseparable_conv2d_9/bias$separable_conv2d_10/depthwise_kernel$separable_conv2d_10/pointwise_kernelseparable_conv2d_10/bias$separable_conv2d_11/depthwise_kernel$separable_conv2d_11/pointwise_kernelseparable_conv2d_11/bias$separable_conv2d_12/depthwise_kernel$separable_conv2d_12/pointwise_kernelseparable_conv2d_12/bias$separable_conv2d_13/depthwise_kernel$separable_conv2d_13/pointwise_kernelseparable_conv2d_13/bias$separable_conv2d_14/depthwise_kernel$separable_conv2d_14/pointwise_kernelseparable_conv2d_14/bias$separable_conv2d_15/depthwise_kernel$separable_conv2d_15/pointwise_kernelseparable_conv2d_15/biasconv2d_2/kernelconv2d_2/biasregression_head_1/kernelregression_head_1/biastotalcountconv2d/kernel/mconv2d/bias/m#separable_conv2d/depthwise_kernel/m#separable_conv2d/pointwise_kernel/mseparable_conv2d/bias/m%separable_conv2d_1/depthwise_kernel/m%separable_conv2d_1/pointwise_kernel/mseparable_conv2d_1/bias/mconv2d_1/kernel/mconv2d_1/bias/m%separable_conv2d_2/depthwise_kernel/m%separable_conv2d_2/pointwise_kernel/mseparable_conv2d_2/bias/m%separable_conv2d_3/depthwise_kernel/m%separable_conv2d_3/pointwise_kernel/mseparable_conv2d_3/bias/m%separable_conv2d_4/depthwise_kernel/m%separable_conv2d_4/pointwise_kernel/mseparable_conv2d_4/bias/m%separable_conv2d_5/depthwise_kernel/m%separable_conv2d_5/pointwise_kernel/mseparable_conv2d_5/bias/m%separable_conv2d_6/depthwise_kernel/m%separable_conv2d_6/pointwise_kernel/mseparable_conv2d_6/bias/m%separable_conv2d_7/depthwise_kernel/m%separable_conv2d_7/pointwise_kernel/mseparable_conv2d_7/bias/m%separable_conv2d_8/depthwise_kernel/m%separable_conv2d_8/pointwise_kernel/mseparable_conv2d_8/bias/m%separable_conv2d_9/depthwise_kernel/m%separable_conv2d_9/pointwise_kernel/mseparable_conv2d_9/bias/m&separable_conv2d_10/depthwise_kernel/m&separable_conv2d_10/pointwise_kernel/mseparable_conv2d_10/bias/m&separable_conv2d_11/depthwise_kernel/m&separable_conv2d_11/pointwise_kernel/mseparable_conv2d_11/bias/m&separable_conv2d_12/depthwise_kernel/m&separable_conv2d_12/pointwise_kernel/mseparable_conv2d_12/bias/m&separable_conv2d_13/depthwise_kernel/m&separable_conv2d_13/pointwise_kernel/mseparable_conv2d_13/bias/m&separable_conv2d_14/depthwise_kernel/m&separable_conv2d_14/pointwise_kernel/mseparable_conv2d_14/bias/m&separable_conv2d_15/depthwise_kernel/m&separable_conv2d_15/pointwise_kernel/mseparable_conv2d_15/bias/mconv2d_2/kernel/mconv2d_2/bias/mregression_head_1/kernel/mregression_head_1/bias/mconv2d/kernel/vconv2d/bias/v#separable_conv2d/depthwise_kernel/v#separable_conv2d/pointwise_kernel/vseparable_conv2d/bias/v%separable_conv2d_1/depthwise_kernel/v%separable_conv2d_1/pointwise_kernel/vseparable_conv2d_1/bias/vconv2d_1/kernel/vconv2d_1/bias/v%separable_conv2d_2/depthwise_kernel/v%separable_conv2d_2/pointwise_kernel/vseparable_conv2d_2/bias/v%separable_conv2d_3/depthwise_kernel/v%separable_conv2d_3/pointwise_kernel/vseparable_conv2d_3/bias/v%separable_conv2d_4/depthwise_kernel/v%separable_conv2d_4/pointwise_kernel/vseparable_conv2d_4/bias/v%separable_conv2d_5/depthwise_kernel/v%separable_conv2d_5/pointwise_kernel/vseparable_conv2d_5/bias/v%separable_conv2d_6/depthwise_kernel/v%separable_conv2d_6/pointwise_kernel/vseparable_conv2d_6/bias/v%separable_conv2d_7/depthwise_kernel/v%separable_conv2d_7/pointwise_kernel/vseparable_conv2d_7/bias/v%separable_conv2d_8/depthwise_kernel/v%separable_conv2d_8/pointwise_kernel/vseparable_conv2d_8/bias/v%separable_conv2d_9/depthwise_kernel/v%separable_conv2d_9/pointwise_kernel/vseparable_conv2d_9/bias/v&separable_conv2d_10/depthwise_kernel/v&separable_conv2d_10/pointwise_kernel/vseparable_conv2d_10/bias/v&separable_conv2d_11/depthwise_kernel/v&separable_conv2d_11/pointwise_kernel/vseparable_conv2d_11/bias/v&separable_conv2d_12/depthwise_kernel/v&separable_conv2d_12/pointwise_kernel/vseparable_conv2d_12/bias/v&separable_conv2d_13/depthwise_kernel/v&separable_conv2d_13/pointwise_kernel/vseparable_conv2d_13/bias/v&separable_conv2d_14/depthwise_kernel/v&separable_conv2d_14/pointwise_kernel/vseparable_conv2d_14/bias/v&separable_conv2d_15/depthwise_kernel/v&separable_conv2d_15/pointwise_kernel/vseparable_conv2d_15/bias/vconv2d_2/kernel/vconv2d_2/bias/vregression_head_1/kernel/vregression_head_1/bias/v*╝
Tin┤
▒2о*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_82574цЖ
б
╫
2__inference_separable_conv2d_2_layer_call_fn_79651

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_796422
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ж
▓
1__inference_regression_head_1_layer_call_fn_81500

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_regression_head_1_layer_call_and_return_conditional_losses_802702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
г
╪
3__inference_separable_conv2d_14_layer_call_fn_79963

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_799542
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
─
й
(__inference_conv2d_2_layer_call_fn_80021

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_800132
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
б
╫
2__inference_separable_conv2d_4_layer_call_fn_79703

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_796942
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_79746

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ё
j
@__inference_add_7_layer_call_and_return_conditional_losses_80250

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:         А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:         А:         А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
╡к
Г2
@__inference_model_layer_call_and_return_conditional_losses_81239

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource=
9separable_conv2d_separable_conv2d_readvariableop_resource?
;separable_conv2d_separable_conv2d_readvariableop_1_resource4
0separable_conv2d_biasadd_readvariableop_resource?
;separable_conv2d_1_separable_conv2d_readvariableop_resourceA
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_1_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource?
;separable_conv2d_2_separable_conv2d_readvariableop_resourceA
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_2_biasadd_readvariableop_resource?
;separable_conv2d_3_separable_conv2d_readvariableop_resourceA
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_3_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource?
;separable_conv2d_6_separable_conv2d_readvariableop_resourceA
=separable_conv2d_6_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_6_biasadd_readvariableop_resource?
;separable_conv2d_7_separable_conv2d_readvariableop_resourceA
=separable_conv2d_7_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_7_biasadd_readvariableop_resource?
;separable_conv2d_8_separable_conv2d_readvariableop_resourceA
=separable_conv2d_8_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_8_biasadd_readvariableop_resource?
;separable_conv2d_9_separable_conv2d_readvariableop_resourceA
=separable_conv2d_9_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_9_biasadd_readvariableop_resource@
<separable_conv2d_10_separable_conv2d_readvariableop_resourceB
>separable_conv2d_10_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_10_biasadd_readvariableop_resource@
<separable_conv2d_11_separable_conv2d_readvariableop_resourceB
>separable_conv2d_11_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_11_biasadd_readvariableop_resource@
<separable_conv2d_12_separable_conv2d_readvariableop_resourceB
>separable_conv2d_12_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_12_biasadd_readvariableop_resource@
<separable_conv2d_13_separable_conv2d_readvariableop_resourceB
>separable_conv2d_13_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_13_biasadd_readvariableop_resource@
<separable_conv2d_14_separable_conv2d_readvariableop_resourceB
>separable_conv2d_14_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_14_biasadd_readvariableop_resource@
<separable_conv2d_15_separable_conv2d_readvariableop_resourceB
>separable_conv2d_15_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_15_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource4
0regression_head_1_matmul_readvariableop_resource5
1regression_head_1_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpв$normalization/Reshape/ReadVariableOpв&normalization/Reshape_1/ReadVariableOpв(regression_head_1/BiasAdd/ReadVariableOpв'regression_head_1/MatMul/ReadVariableOpв'separable_conv2d/BiasAdd/ReadVariableOpв0separable_conv2d/separable_conv2d/ReadVariableOpв2separable_conv2d/separable_conv2d/ReadVariableOp_1в)separable_conv2d_1/BiasAdd/ReadVariableOpв2separable_conv2d_1/separable_conv2d/ReadVariableOpв4separable_conv2d_1/separable_conv2d/ReadVariableOp_1в*separable_conv2d_10/BiasAdd/ReadVariableOpв3separable_conv2d_10/separable_conv2d/ReadVariableOpв5separable_conv2d_10/separable_conv2d/ReadVariableOp_1в*separable_conv2d_11/BiasAdd/ReadVariableOpв3separable_conv2d_11/separable_conv2d/ReadVariableOpв5separable_conv2d_11/separable_conv2d/ReadVariableOp_1в*separable_conv2d_12/BiasAdd/ReadVariableOpв3separable_conv2d_12/separable_conv2d/ReadVariableOpв5separable_conv2d_12/separable_conv2d/ReadVariableOp_1в*separable_conv2d_13/BiasAdd/ReadVariableOpв3separable_conv2d_13/separable_conv2d/ReadVariableOpв5separable_conv2d_13/separable_conv2d/ReadVariableOp_1в*separable_conv2d_14/BiasAdd/ReadVariableOpв3separable_conv2d_14/separable_conv2d/ReadVariableOpв5separable_conv2d_14/separable_conv2d/ReadVariableOp_1в*separable_conv2d_15/BiasAdd/ReadVariableOpв3separable_conv2d_15/separable_conv2d/ReadVariableOpв5separable_conv2d_15/separable_conv2d/ReadVariableOp_1в)separable_conv2d_2/BiasAdd/ReadVariableOpв2separable_conv2d_2/separable_conv2d/ReadVariableOpв4separable_conv2d_2/separable_conv2d/ReadVariableOp_1в)separable_conv2d_3/BiasAdd/ReadVariableOpв2separable_conv2d_3/separable_conv2d/ReadVariableOpв4separable_conv2d_3/separable_conv2d/ReadVariableOp_1в)separable_conv2d_4/BiasAdd/ReadVariableOpв2separable_conv2d_4/separable_conv2d/ReadVariableOpв4separable_conv2d_4/separable_conv2d/ReadVariableOp_1в)separable_conv2d_5/BiasAdd/ReadVariableOpв2separable_conv2d_5/separable_conv2d/ReadVariableOpв4separable_conv2d_5/separable_conv2d/ReadVariableOp_1в)separable_conv2d_6/BiasAdd/ReadVariableOpв2separable_conv2d_6/separable_conv2d/ReadVariableOpв4separable_conv2d_6/separable_conv2d/ReadVariableOp_1в)separable_conv2d_7/BiasAdd/ReadVariableOpв2separable_conv2d_7/separable_conv2d/ReadVariableOpв4separable_conv2d_7/separable_conv2d/ReadVariableOp_1в)separable_conv2d_8/BiasAdd/ReadVariableOpв2separable_conv2d_8/separable_conv2d/ReadVariableOpв4separable_conv2d_8/separable_conv2d/ReadVariableOp_1в)separable_conv2d_9/BiasAdd/ReadVariableOpв2separable_conv2d_9/separable_conv2d/ReadVariableOpв4separable_conv2d_9/separable_conv2d/ReadVariableOp_1╢
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOpУ
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape╛
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape╝
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOpЧ
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape╞
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1П
normalization/subSubinputsnormalization/Reshape:output:0*
T0*/
_output_shapes
:         @@2
normalization/subГ
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtв
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*/
_output_shapes
:         @@2
normalization/truedivк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╦
conv2d/Conv2DConv2Dnormalization/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:           @2
conv2d/Seluц
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpэ
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1л
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2)
'separable_conv2d/separable_conv2d/Shape│
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rateк
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwiseж
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d└
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp╫
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d/BiasAddФ
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d/Seluэ
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_1/separable_conv2d/Shape╖
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rate╗
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwiseо
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d╞
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOp▀
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_1/BiasAddЪ
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_1/Selu▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╥
conv2d_1/Conv2DConv2Dconv2d/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpн
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
conv2d_1/BiasAddШ
add/addAddV2%separable_conv2d_1/Selu:activations:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:           А2	
add/addэ
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_2/separable_conv2d/Shape╖
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rateг
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativeadd/add:z:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwiseо
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2d╞
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOp▀
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_2/BiasAddЪ
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_2/Seluэ
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_3/separable_conv2d/Shape╖
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rate╜
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_2/Selu:activations:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwiseо
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2d╞
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOp▀
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_3/BiasAddЪ
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_3/SeluО
	add_1/addAddV2%separable_conv2d_3/Selu:activations:0add/add:z:0*
T0*0
_output_shapes
:           А2
	add_1/addэ
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_4/separable_conv2d/Shape╖
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rateе
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_1/add:z:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwiseо
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2d╞
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOp▀
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_4/BiasAddЪ
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_4/Seluэ
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_5/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_5/separable_conv2d/Shape╖
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_5/separable_conv2d/dilation_rate╜
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_4/Selu:activations:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_5/separable_conv2d/depthwiseо
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_5/separable_conv2d╞
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_5/BiasAdd/ReadVariableOp▀
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_5/BiasAddЪ
separable_conv2d_5/SeluSelu#separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_5/SeluР
	add_2/addAddV2%separable_conv2d_5/Selu:activations:0add_1/add:z:0*
T0*0
_output_shapes
:           А2
	add_2/addэ
2separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_6_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_6/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_6_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_6/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_6/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_6/separable_conv2d/Shape╖
1separable_conv2d_6/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_6/separable_conv2d/dilation_rateе
-separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_2/add:z:0:separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_6/separable_conv2d/depthwiseо
#separable_conv2d_6/separable_conv2dConv2D6separable_conv2d_6/separable_conv2d/depthwise:output:0<separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_6/separable_conv2d╞
)separable_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_6/BiasAdd/ReadVariableOp▀
separable_conv2d_6/BiasAddBiasAdd,separable_conv2d_6/separable_conv2d:output:01separable_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_6/BiasAddЪ
separable_conv2d_6/SeluSelu#separable_conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_6/Seluэ
2separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_7_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_7/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_7_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_7/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_7/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_7/separable_conv2d/Shape╖
1separable_conv2d_7/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_7/separable_conv2d/dilation_rate╜
-separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_6/Selu:activations:0:separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_7/separable_conv2d/depthwiseо
#separable_conv2d_7/separable_conv2dConv2D6separable_conv2d_7/separable_conv2d/depthwise:output:0<separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_7/separable_conv2d╞
)separable_conv2d_7/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_7/BiasAdd/ReadVariableOp▀
separable_conv2d_7/BiasAddBiasAdd,separable_conv2d_7/separable_conv2d:output:01separable_conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_7/BiasAddЪ
separable_conv2d_7/SeluSelu#separable_conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_7/SeluР
	add_3/addAddV2%separable_conv2d_7/Selu:activations:0add_2/add:z:0*
T0*0
_output_shapes
:           А2
	add_3/addэ
2separable_conv2d_8/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_8_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_8/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_8/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_8_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_8/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_8/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_8/separable_conv2d/Shape╖
1separable_conv2d_8/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_8/separable_conv2d/dilation_rateе
-separable_conv2d_8/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_3/add:z:0:separable_conv2d_8/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_8/separable_conv2d/depthwiseо
#separable_conv2d_8/separable_conv2dConv2D6separable_conv2d_8/separable_conv2d/depthwise:output:0<separable_conv2d_8/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_8/separable_conv2d╞
)separable_conv2d_8/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_8/BiasAdd/ReadVariableOp▀
separable_conv2d_8/BiasAddBiasAdd,separable_conv2d_8/separable_conv2d:output:01separable_conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_8/BiasAddЪ
separable_conv2d_8/SeluSelu#separable_conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_8/Seluэ
2separable_conv2d_9/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_9_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_9/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_9/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_9_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_9/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_9/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_9/separable_conv2d/Shape╖
1separable_conv2d_9/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_9/separable_conv2d/dilation_rate╜
-separable_conv2d_9/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_8/Selu:activations:0:separable_conv2d_9/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_9/separable_conv2d/depthwiseо
#separable_conv2d_9/separable_conv2dConv2D6separable_conv2d_9/separable_conv2d/depthwise:output:0<separable_conv2d_9/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_9/separable_conv2d╞
)separable_conv2d_9/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_9/BiasAdd/ReadVariableOp▀
separable_conv2d_9/BiasAddBiasAdd,separable_conv2d_9/separable_conv2d:output:01separable_conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_9/BiasAddЪ
separable_conv2d_9/SeluSelu#separable_conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_9/SeluР
	add_4/addAddV2%separable_conv2d_9/Selu:activations:0add_3/add:z:0*
T0*0
_output_shapes
:           А2
	add_4/addЁ
3separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_10_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_10/separable_conv2d/ReadVariableOpў
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_10_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_10/separable_conv2d/Shape╣
2separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_10/separable_conv2d/dilation_rateи
.separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_4/add:z:0;separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_10/separable_conv2d/depthwise▓
$separable_conv2d_10/separable_conv2dConv2D7separable_conv2d_10/separable_conv2d/depthwise:output:0=separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_10/separable_conv2d╔
*separable_conv2d_10/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_10/BiasAdd/ReadVariableOpу
separable_conv2d_10/BiasAddBiasAdd-separable_conv2d_10/separable_conv2d:output:02separable_conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_10/BiasAddЭ
separable_conv2d_10/SeluSelu$separable_conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_10/SeluЁ
3separable_conv2d_11/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_11_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_11/separable_conv2d/ReadVariableOpў
5separable_conv2d_11/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_11_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_11/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_11/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_11/separable_conv2d/Shape╣
2separable_conv2d_11/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_11/separable_conv2d/dilation_rate┴
.separable_conv2d_11/separable_conv2d/depthwiseDepthwiseConv2dNative&separable_conv2d_10/Selu:activations:0;separable_conv2d_11/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_11/separable_conv2d/depthwise▓
$separable_conv2d_11/separable_conv2dConv2D7separable_conv2d_11/separable_conv2d/depthwise:output:0=separable_conv2d_11/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_11/separable_conv2d╔
*separable_conv2d_11/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_11/BiasAdd/ReadVariableOpу
separable_conv2d_11/BiasAddBiasAdd-separable_conv2d_11/separable_conv2d:output:02separable_conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_11/BiasAddЭ
separable_conv2d_11/SeluSelu$separable_conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_11/SeluС
	add_5/addAddV2&separable_conv2d_11/Selu:activations:0add_4/add:z:0*
T0*0
_output_shapes
:           А2
	add_5/addЁ
3separable_conv2d_12/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_12_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_12/separable_conv2d/ReadVariableOpў
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_12_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_12/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_12/separable_conv2d/Shape╣
2separable_conv2d_12/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_12/separable_conv2d/dilation_rateи
.separable_conv2d_12/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_5/add:z:0;separable_conv2d_12/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_12/separable_conv2d/depthwise▓
$separable_conv2d_12/separable_conv2dConv2D7separable_conv2d_12/separable_conv2d/depthwise:output:0=separable_conv2d_12/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_12/separable_conv2d╔
*separable_conv2d_12/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_12/BiasAdd/ReadVariableOpу
separable_conv2d_12/BiasAddBiasAdd-separable_conv2d_12/separable_conv2d:output:02separable_conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_12/BiasAddЭ
separable_conv2d_12/SeluSelu$separable_conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_12/SeluЁ
3separable_conv2d_13/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_13_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_13/separable_conv2d/ReadVariableOpў
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_13_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_13/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_13/separable_conv2d/Shape╣
2separable_conv2d_13/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_13/separable_conv2d/dilation_rate┴
.separable_conv2d_13/separable_conv2d/depthwiseDepthwiseConv2dNative&separable_conv2d_12/Selu:activations:0;separable_conv2d_13/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_13/separable_conv2d/depthwise▓
$separable_conv2d_13/separable_conv2dConv2D7separable_conv2d_13/separable_conv2d/depthwise:output:0=separable_conv2d_13/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_13/separable_conv2d╔
*separable_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_13/BiasAdd/ReadVariableOpу
separable_conv2d_13/BiasAddBiasAdd-separable_conv2d_13/separable_conv2d:output:02separable_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_13/BiasAddЭ
separable_conv2d_13/SeluSelu$separable_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_13/SeluС
	add_6/addAddV2&separable_conv2d_13/Selu:activations:0add_5/add:z:0*
T0*0
_output_shapes
:           А2
	add_6/addЁ
3separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_14_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_14/separable_conv2d/ReadVariableOpў
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_14_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_14/separable_conv2d/Shape╣
2separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_14/separable_conv2d/dilation_rateи
.separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_6/add:z:0;separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_14/separable_conv2d/depthwise▓
$separable_conv2d_14/separable_conv2dConv2D7separable_conv2d_14/separable_conv2d/depthwise:output:0=separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_14/separable_conv2d╔
*separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_14/BiasAdd/ReadVariableOpу
separable_conv2d_14/BiasAddBiasAdd-separable_conv2d_14/separable_conv2d:output:02separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_14/BiasAddЭ
separable_conv2d_14/SeluSelu$separable_conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_14/SeluЁ
3separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_15_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_15/separable_conv2d/ReadVariableOpў
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_15_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_15/separable_conv2d/Shape╣
2separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_15/separable_conv2d/dilation_rate┴
.separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative&separable_conv2d_14/Selu:activations:0;separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_15/separable_conv2d/depthwise▓
$separable_conv2d_15/separable_conv2dConv2D7separable_conv2d_15/separable_conv2d/depthwise:output:0=separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_15/separable_conv2d╔
*separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_15/BiasAdd/ReadVariableOpу
separable_conv2d_15/BiasAddBiasAdd-separable_conv2d_15/separable_conv2d:output:02separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_15/BiasAddЭ
separable_conv2d_15/SeluSelu$separable_conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_15/Selu╬
max_pooling2d/MaxPoolMaxPool&separable_conv2d_15/Selu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool▓
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╞
conv2d_2/Conv2DConv2Dadd_6/add:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_2/BiasAddХ
	add_7/addAddV2max_pooling2d/MaxPool:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
	add_7/add│
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices┬
global_average_pooling2d/MeanMeanadd_7/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
global_average_pooling2d/Mean─
'regression_head_1/MatMul/ReadVariableOpReadVariableOp0regression_head_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'regression_head_1/MatMul/ReadVariableOp╔
regression_head_1/MatMulMatMul&global_average_pooling2d/Mean:output:0/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
regression_head_1/MatMul┬
(regression_head_1/BiasAdd/ReadVariableOpReadVariableOp1regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(regression_head_1/BiasAdd/ReadVariableOp╔
regression_head_1/BiasAddBiasAdd"regression_head_1/MatMul:product:00regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
regression_head_1/BiasAddь
IdentityIdentity"regression_head_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp)^regression_head_1/BiasAdd/ReadVariableOp(^regression_head_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1+^separable_conv2d_10/BiasAdd/ReadVariableOp4^separable_conv2d_10/separable_conv2d/ReadVariableOp6^separable_conv2d_10/separable_conv2d/ReadVariableOp_1+^separable_conv2d_11/BiasAdd/ReadVariableOp4^separable_conv2d_11/separable_conv2d/ReadVariableOp6^separable_conv2d_11/separable_conv2d/ReadVariableOp_1+^separable_conv2d_12/BiasAdd/ReadVariableOp4^separable_conv2d_12/separable_conv2d/ReadVariableOp6^separable_conv2d_12/separable_conv2d/ReadVariableOp_1+^separable_conv2d_13/BiasAdd/ReadVariableOp4^separable_conv2d_13/separable_conv2d/ReadVariableOp6^separable_conv2d_13/separable_conv2d/ReadVariableOp_1+^separable_conv2d_14/BiasAdd/ReadVariableOp4^separable_conv2d_14/separable_conv2d/ReadVariableOp6^separable_conv2d_14/separable_conv2d/ReadVariableOp_1+^separable_conv2d_15/BiasAdd/ReadVariableOp4^separable_conv2d_15/separable_conv2d/ReadVariableOp6^separable_conv2d_15/separable_conv2d/ReadVariableOp_1*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*^separable_conv2d_5/BiasAdd/ReadVariableOp3^separable_conv2d_5/separable_conv2d/ReadVariableOp5^separable_conv2d_5/separable_conv2d/ReadVariableOp_1*^separable_conv2d_6/BiasAdd/ReadVariableOp3^separable_conv2d_6/separable_conv2d/ReadVariableOp5^separable_conv2d_6/separable_conv2d/ReadVariableOp_1*^separable_conv2d_7/BiasAdd/ReadVariableOp3^separable_conv2d_7/separable_conv2d/ReadVariableOp5^separable_conv2d_7/separable_conv2d/ReadVariableOp_1*^separable_conv2d_8/BiasAdd/ReadVariableOp3^separable_conv2d_8/separable_conv2d/ReadVariableOp5^separable_conv2d_8/separable_conv2d/ReadVariableOp_1*^separable_conv2d_9/BiasAdd/ReadVariableOp3^separable_conv2d_9/separable_conv2d/ReadVariableOp5^separable_conv2d_9/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2T
(regression_head_1/BiasAdd/ReadVariableOp(regression_head_1/BiasAdd/ReadVariableOp2R
'regression_head_1/MatMul/ReadVariableOp'regression_head_1/MatMul/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_10/BiasAdd/ReadVariableOp*separable_conv2d_10/BiasAdd/ReadVariableOp2j
3separable_conv2d_10/separable_conv2d/ReadVariableOp3separable_conv2d_10/separable_conv2d/ReadVariableOp2n
5separable_conv2d_10/separable_conv2d/ReadVariableOp_15separable_conv2d_10/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_11/BiasAdd/ReadVariableOp*separable_conv2d_11/BiasAdd/ReadVariableOp2j
3separable_conv2d_11/separable_conv2d/ReadVariableOp3separable_conv2d_11/separable_conv2d/ReadVariableOp2n
5separable_conv2d_11/separable_conv2d/ReadVariableOp_15separable_conv2d_11/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_12/BiasAdd/ReadVariableOp*separable_conv2d_12/BiasAdd/ReadVariableOp2j
3separable_conv2d_12/separable_conv2d/ReadVariableOp3separable_conv2d_12/separable_conv2d/ReadVariableOp2n
5separable_conv2d_12/separable_conv2d/ReadVariableOp_15separable_conv2d_12/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_13/BiasAdd/ReadVariableOp*separable_conv2d_13/BiasAdd/ReadVariableOp2j
3separable_conv2d_13/separable_conv2d/ReadVariableOp3separable_conv2d_13/separable_conv2d/ReadVariableOp2n
5separable_conv2d_13/separable_conv2d/ReadVariableOp_15separable_conv2d_13/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_14/BiasAdd/ReadVariableOp*separable_conv2d_14/BiasAdd/ReadVariableOp2j
3separable_conv2d_14/separable_conv2d/ReadVariableOp3separable_conv2d_14/separable_conv2d/ReadVariableOp2n
5separable_conv2d_14/separable_conv2d/ReadVariableOp_15separable_conv2d_14/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_15/BiasAdd/ReadVariableOp*separable_conv2d_15/BiasAdd/ReadVariableOp2j
3separable_conv2d_15/separable_conv2d/ReadVariableOp3separable_conv2d_15/separable_conv2d/ReadVariableOp2n
5separable_conv2d_15/separable_conv2d/ReadVariableOp_15separable_conv2d_15/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_5/BiasAdd/ReadVariableOp)separable_conv2d_5/BiasAdd/ReadVariableOp2h
2separable_conv2d_5/separable_conv2d/ReadVariableOp2separable_conv2d_5/separable_conv2d/ReadVariableOp2l
4separable_conv2d_5/separable_conv2d/ReadVariableOp_14separable_conv2d_5/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_6/BiasAdd/ReadVariableOp)separable_conv2d_6/BiasAdd/ReadVariableOp2h
2separable_conv2d_6/separable_conv2d/ReadVariableOp2separable_conv2d_6/separable_conv2d/ReadVariableOp2l
4separable_conv2d_6/separable_conv2d/ReadVariableOp_14separable_conv2d_6/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_7/BiasAdd/ReadVariableOp)separable_conv2d_7/BiasAdd/ReadVariableOp2h
2separable_conv2d_7/separable_conv2d/ReadVariableOp2separable_conv2d_7/separable_conv2d/ReadVariableOp2l
4separable_conv2d_7/separable_conv2d/ReadVariableOp_14separable_conv2d_7/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_8/BiasAdd/ReadVariableOp)separable_conv2d_8/BiasAdd/ReadVariableOp2h
2separable_conv2d_8/separable_conv2d/ReadVariableOp2separable_conv2d_8/separable_conv2d/ReadVariableOp2l
4separable_conv2d_8/separable_conv2d/ReadVariableOp_14separable_conv2d_8/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_9/BiasAdd/ReadVariableOp)separable_conv2d_9/BiasAdd/ReadVariableOp2h
2separable_conv2d_9/separable_conv2d/ReadVariableOp2separable_conv2d_9/separable_conv2d/ReadVariableOp2l
4separable_conv2d_9/separable_conv2d/ReadVariableOp_14separable_conv2d_9/separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
г
╪
3__inference_separable_conv2d_11_layer_call_fn_79885

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_798762
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_79798

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
│'
╢
#__inference_signature_wrapper_80753
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54#
statefulpartitionedcall_args_55#
statefulpartitionedcall_args_56#
statefulpartitionedcall_args_57#
statefulpartitionedcall_args_58
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54statefulpartitionedcall_args_55statefulpartitionedcall_args_56statefulpartitionedcall_args_57statefulpartitionedcall_args_58*F
Tin?
=2;*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_795322
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ь
╒
0__inference_separable_conv2d_layer_call_fn_79579

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_795702
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ё
j
@__inference_add_4_layer_call_and_return_conditional_losses_80177

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
°
l
@__inference_add_1_layer_call_and_return_conditional_losses_81405
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
х
┌
A__inference_conv2d_layer_call_and_return_conditional_losses_79545

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Selu▒
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
г
╪
3__inference_separable_conv2d_13_layer_call_fn_79937

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_799282
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╒'
╕
%__inference_model_layer_call_fn_80689
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54#
statefulpartitionedcall_args_55#
statefulpartitionedcall_args_56#
statefulpartitionedcall_args_57#
statefulpartitionedcall_args_58
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54statefulpartitionedcall_args_55statefulpartitionedcall_args_56statefulpartitionedcall_args_57statefulpartitionedcall_args_58*F
Tin?
=2;*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_806282
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
┘╛
БV
__inference__traced_save_82043
file_prefix1
-savev2_normalization_mean_read_readvariableop5
1savev2_normalization_variance_read_readvariableop2
.savev2_normalization_count_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopB
>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_2_bias_read_readvariableopB
>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_3_bias_read_readvariableopB
>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_4_bias_read_readvariableopB
>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_5_bias_read_readvariableopB
>savev2_separable_conv2d_6_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_6_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_6_bias_read_readvariableopB
>savev2_separable_conv2d_7_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_7_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_7_bias_read_readvariableopB
>savev2_separable_conv2d_8_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_8_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_8_bias_read_readvariableopB
>savev2_separable_conv2d_9_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_9_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_9_bias_read_readvariableopC
?savev2_separable_conv2d_10_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_10_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_10_bias_read_readvariableopC
?savev2_separable_conv2d_11_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_11_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_11_bias_read_readvariableopC
?savev2_separable_conv2d_12_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_12_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_12_bias_read_readvariableopC
?savev2_separable_conv2d_13_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_13_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_13_bias_read_readvariableopC
?savev2_separable_conv2d_14_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_14_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_14_bias_read_readvariableopC
?savev2_separable_conv2d_15_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_15_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_15_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop7
3savev2_regression_head_1_kernel_read_readvariableop5
1savev2_regression_head_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_conv2d_kernel_m_read_readvariableop,
(savev2_conv2d_bias_m_read_readvariableopB
>savev2_separable_conv2d_depthwise_kernel_m_read_readvariableopB
>savev2_separable_conv2d_pointwise_kernel_m_read_readvariableop6
2savev2_separable_conv2d_bias_m_read_readvariableopD
@savev2_separable_conv2d_1_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_1_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_1_bias_m_read_readvariableop0
,savev2_conv2d_1_kernel_m_read_readvariableop.
*savev2_conv2d_1_bias_m_read_readvariableopD
@savev2_separable_conv2d_2_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_2_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_2_bias_m_read_readvariableopD
@savev2_separable_conv2d_3_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_3_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_3_bias_m_read_readvariableopD
@savev2_separable_conv2d_4_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_4_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_4_bias_m_read_readvariableopD
@savev2_separable_conv2d_5_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_5_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_5_bias_m_read_readvariableopD
@savev2_separable_conv2d_6_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_6_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_6_bias_m_read_readvariableopD
@savev2_separable_conv2d_7_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_7_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_7_bias_m_read_readvariableopD
@savev2_separable_conv2d_8_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_8_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_8_bias_m_read_readvariableopD
@savev2_separable_conv2d_9_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_9_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_9_bias_m_read_readvariableopE
Asavev2_separable_conv2d_10_depthwise_kernel_m_read_readvariableopE
Asavev2_separable_conv2d_10_pointwise_kernel_m_read_readvariableop9
5savev2_separable_conv2d_10_bias_m_read_readvariableopE
Asavev2_separable_conv2d_11_depthwise_kernel_m_read_readvariableopE
Asavev2_separable_conv2d_11_pointwise_kernel_m_read_readvariableop9
5savev2_separable_conv2d_11_bias_m_read_readvariableopE
Asavev2_separable_conv2d_12_depthwise_kernel_m_read_readvariableopE
Asavev2_separable_conv2d_12_pointwise_kernel_m_read_readvariableop9
5savev2_separable_conv2d_12_bias_m_read_readvariableopE
Asavev2_separable_conv2d_13_depthwise_kernel_m_read_readvariableopE
Asavev2_separable_conv2d_13_pointwise_kernel_m_read_readvariableop9
5savev2_separable_conv2d_13_bias_m_read_readvariableopE
Asavev2_separable_conv2d_14_depthwise_kernel_m_read_readvariableopE
Asavev2_separable_conv2d_14_pointwise_kernel_m_read_readvariableop9
5savev2_separable_conv2d_14_bias_m_read_readvariableopE
Asavev2_separable_conv2d_15_depthwise_kernel_m_read_readvariableopE
Asavev2_separable_conv2d_15_pointwise_kernel_m_read_readvariableop9
5savev2_separable_conv2d_15_bias_m_read_readvariableop0
,savev2_conv2d_2_kernel_m_read_readvariableop.
*savev2_conv2d_2_bias_m_read_readvariableop9
5savev2_regression_head_1_kernel_m_read_readvariableop7
3savev2_regression_head_1_bias_m_read_readvariableop.
*savev2_conv2d_kernel_v_read_readvariableop,
(savev2_conv2d_bias_v_read_readvariableopB
>savev2_separable_conv2d_depthwise_kernel_v_read_readvariableopB
>savev2_separable_conv2d_pointwise_kernel_v_read_readvariableop6
2savev2_separable_conv2d_bias_v_read_readvariableopD
@savev2_separable_conv2d_1_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_1_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_1_bias_v_read_readvariableop0
,savev2_conv2d_1_kernel_v_read_readvariableop.
*savev2_conv2d_1_bias_v_read_readvariableopD
@savev2_separable_conv2d_2_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_2_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_2_bias_v_read_readvariableopD
@savev2_separable_conv2d_3_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_3_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_3_bias_v_read_readvariableopD
@savev2_separable_conv2d_4_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_4_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_4_bias_v_read_readvariableopD
@savev2_separable_conv2d_5_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_5_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_5_bias_v_read_readvariableopD
@savev2_separable_conv2d_6_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_6_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_6_bias_v_read_readvariableopD
@savev2_separable_conv2d_7_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_7_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_7_bias_v_read_readvariableopD
@savev2_separable_conv2d_8_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_8_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_8_bias_v_read_readvariableopD
@savev2_separable_conv2d_9_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_9_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_9_bias_v_read_readvariableopE
Asavev2_separable_conv2d_10_depthwise_kernel_v_read_readvariableopE
Asavev2_separable_conv2d_10_pointwise_kernel_v_read_readvariableop9
5savev2_separable_conv2d_10_bias_v_read_readvariableopE
Asavev2_separable_conv2d_11_depthwise_kernel_v_read_readvariableopE
Asavev2_separable_conv2d_11_pointwise_kernel_v_read_readvariableop9
5savev2_separable_conv2d_11_bias_v_read_readvariableopE
Asavev2_separable_conv2d_12_depthwise_kernel_v_read_readvariableopE
Asavev2_separable_conv2d_12_pointwise_kernel_v_read_readvariableop9
5savev2_separable_conv2d_12_bias_v_read_readvariableopE
Asavev2_separable_conv2d_13_depthwise_kernel_v_read_readvariableopE
Asavev2_separable_conv2d_13_pointwise_kernel_v_read_readvariableop9
5savev2_separable_conv2d_13_bias_v_read_readvariableopE
Asavev2_separable_conv2d_14_depthwise_kernel_v_read_readvariableopE
Asavev2_separable_conv2d_14_pointwise_kernel_v_read_readvariableop9
5savev2_separable_conv2d_14_bias_v_read_readvariableopE
Asavev2_separable_conv2d_15_depthwise_kernel_v_read_readvariableopE
Asavev2_separable_conv2d_15_pointwise_kernel_v_read_readvariableop9
5savev2_separable_conv2d_15_bias_v_read_readvariableop0
,savev2_conv2d_2_kernel_v_read_readvariableop.
*savev2_conv2d_2_bias_v_read_readvariableop9
5savev2_regression_head_1_kernel_v_read_readvariableop7
3savev2_regression_head_1_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4f88efcdcd6744f59d1500723b1fdb7f/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╜l
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:н*
dtype0*╬k
value─kB┴kнB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesч
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:н*
dtype0*Ё
valueцBунB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesиR
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_normalization_mean_read_readvariableop1savev2_normalization_variance_read_readvariableop.savev2_normalization_count_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_5_bias_read_readvariableop>savev2_separable_conv2d_6_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_6_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_6_bias_read_readvariableop>savev2_separable_conv2d_7_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_7_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_7_bias_read_readvariableop>savev2_separable_conv2d_8_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_8_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_8_bias_read_readvariableop>savev2_separable_conv2d_9_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_9_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_9_bias_read_readvariableop?savev2_separable_conv2d_10_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_10_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_10_bias_read_readvariableop?savev2_separable_conv2d_11_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_11_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_11_bias_read_readvariableop?savev2_separable_conv2d_12_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_12_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_12_bias_read_readvariableop?savev2_separable_conv2d_13_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_13_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_13_bias_read_readvariableop?savev2_separable_conv2d_14_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_14_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_14_bias_read_readvariableop?savev2_separable_conv2d_15_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_15_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_15_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop3savev2_regression_head_1_kernel_read_readvariableop1savev2_regression_head_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_conv2d_kernel_m_read_readvariableop(savev2_conv2d_bias_m_read_readvariableop>savev2_separable_conv2d_depthwise_kernel_m_read_readvariableop>savev2_separable_conv2d_pointwise_kernel_m_read_readvariableop2savev2_separable_conv2d_bias_m_read_readvariableop@savev2_separable_conv2d_1_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_1_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_1_bias_m_read_readvariableop,savev2_conv2d_1_kernel_m_read_readvariableop*savev2_conv2d_1_bias_m_read_readvariableop@savev2_separable_conv2d_2_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_2_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_2_bias_m_read_readvariableop@savev2_separable_conv2d_3_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_3_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_3_bias_m_read_readvariableop@savev2_separable_conv2d_4_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_4_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_4_bias_m_read_readvariableop@savev2_separable_conv2d_5_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_5_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_5_bias_m_read_readvariableop@savev2_separable_conv2d_6_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_6_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_6_bias_m_read_readvariableop@savev2_separable_conv2d_7_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_7_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_7_bias_m_read_readvariableop@savev2_separable_conv2d_8_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_8_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_8_bias_m_read_readvariableop@savev2_separable_conv2d_9_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_9_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_9_bias_m_read_readvariableopAsavev2_separable_conv2d_10_depthwise_kernel_m_read_readvariableopAsavev2_separable_conv2d_10_pointwise_kernel_m_read_readvariableop5savev2_separable_conv2d_10_bias_m_read_readvariableopAsavev2_separable_conv2d_11_depthwise_kernel_m_read_readvariableopAsavev2_separable_conv2d_11_pointwise_kernel_m_read_readvariableop5savev2_separable_conv2d_11_bias_m_read_readvariableopAsavev2_separable_conv2d_12_depthwise_kernel_m_read_readvariableopAsavev2_separable_conv2d_12_pointwise_kernel_m_read_readvariableop5savev2_separable_conv2d_12_bias_m_read_readvariableopAsavev2_separable_conv2d_13_depthwise_kernel_m_read_readvariableopAsavev2_separable_conv2d_13_pointwise_kernel_m_read_readvariableop5savev2_separable_conv2d_13_bias_m_read_readvariableopAsavev2_separable_conv2d_14_depthwise_kernel_m_read_readvariableopAsavev2_separable_conv2d_14_pointwise_kernel_m_read_readvariableop5savev2_separable_conv2d_14_bias_m_read_readvariableopAsavev2_separable_conv2d_15_depthwise_kernel_m_read_readvariableopAsavev2_separable_conv2d_15_pointwise_kernel_m_read_readvariableop5savev2_separable_conv2d_15_bias_m_read_readvariableop,savev2_conv2d_2_kernel_m_read_readvariableop*savev2_conv2d_2_bias_m_read_readvariableop5savev2_regression_head_1_kernel_m_read_readvariableop3savev2_regression_head_1_bias_m_read_readvariableop*savev2_conv2d_kernel_v_read_readvariableop(savev2_conv2d_bias_v_read_readvariableop>savev2_separable_conv2d_depthwise_kernel_v_read_readvariableop>savev2_separable_conv2d_pointwise_kernel_v_read_readvariableop2savev2_separable_conv2d_bias_v_read_readvariableop@savev2_separable_conv2d_1_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_1_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_1_bias_v_read_readvariableop,savev2_conv2d_1_kernel_v_read_readvariableop*savev2_conv2d_1_bias_v_read_readvariableop@savev2_separable_conv2d_2_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_2_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_2_bias_v_read_readvariableop@savev2_separable_conv2d_3_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_3_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_3_bias_v_read_readvariableop@savev2_separable_conv2d_4_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_4_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_4_bias_v_read_readvariableop@savev2_separable_conv2d_5_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_5_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_5_bias_v_read_readvariableop@savev2_separable_conv2d_6_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_6_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_6_bias_v_read_readvariableop@savev2_separable_conv2d_7_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_7_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_7_bias_v_read_readvariableop@savev2_separable_conv2d_8_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_8_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_8_bias_v_read_readvariableop@savev2_separable_conv2d_9_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_9_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_9_bias_v_read_readvariableopAsavev2_separable_conv2d_10_depthwise_kernel_v_read_readvariableopAsavev2_separable_conv2d_10_pointwise_kernel_v_read_readvariableop5savev2_separable_conv2d_10_bias_v_read_readvariableopAsavev2_separable_conv2d_11_depthwise_kernel_v_read_readvariableopAsavev2_separable_conv2d_11_pointwise_kernel_v_read_readvariableop5savev2_separable_conv2d_11_bias_v_read_readvariableopAsavev2_separable_conv2d_12_depthwise_kernel_v_read_readvariableopAsavev2_separable_conv2d_12_pointwise_kernel_v_read_readvariableop5savev2_separable_conv2d_12_bias_v_read_readvariableopAsavev2_separable_conv2d_13_depthwise_kernel_v_read_readvariableopAsavev2_separable_conv2d_13_pointwise_kernel_v_read_readvariableop5savev2_separable_conv2d_13_bias_v_read_readvariableopAsavev2_separable_conv2d_14_depthwise_kernel_v_read_readvariableopAsavev2_separable_conv2d_14_pointwise_kernel_v_read_readvariableop5savev2_separable_conv2d_14_bias_v_read_readvariableopAsavev2_separable_conv2d_15_depthwise_kernel_v_read_readvariableopAsavev2_separable_conv2d_15_pointwise_kernel_v_read_readvariableop5savev2_separable_conv2d_15_bias_v_read_readvariableop,savev2_conv2d_2_kernel_v_read_readvariableop*savev2_conv2d_2_bias_v_read_readvariableop5savev2_regression_head_1_kernel_v_read_readvariableop3savev2_regression_head_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *╛
dtypes│
░2н2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*▀
_input_shapes═
╩: ::: :@:@:@:@А:А:А:АА:А:@А:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:АА:А:	А:: : :@:@:@:@А:А:А:АА:А:@А:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:АА:А:	А::@:@:@:@А:А:А:АА:А:@А:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
▒
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_79995

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_79772

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
╥'
╖
%__inference_model_layer_call_fn_81365

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54#
statefulpartitionedcall_args_55#
statefulpartitionedcall_args_56#
statefulpartitionedcall_args_57#
statefulpartitionedcall_args_58
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54statefulpartitionedcall_args_55statefulpartitionedcall_args_56statefulpartitionedcall_args_57statefulpartitionedcall_args_58*F
Tin?
=2;*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_806282
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╬
Q
%__inference_add_5_layer_call_fn_81459
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_802002
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╜
з
&__inference_conv2d_layer_call_fn_79553

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_795452
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
г
╪
3__inference_separable_conv2d_15_layer_call_fn_79989

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_799802
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
▄│
д 
@__inference_model_layer_call_and_return_conditional_losses_80472

inputs0
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_35
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_35
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_35
1separable_conv2d_8_statefulpartitionedcall_args_15
1separable_conv2d_8_statefulpartitionedcall_args_25
1separable_conv2d_8_statefulpartitionedcall_args_35
1separable_conv2d_9_statefulpartitionedcall_args_15
1separable_conv2d_9_statefulpartitionedcall_args_25
1separable_conv2d_9_statefulpartitionedcall_args_36
2separable_conv2d_10_statefulpartitionedcall_args_16
2separable_conv2d_10_statefulpartitionedcall_args_26
2separable_conv2d_10_statefulpartitionedcall_args_36
2separable_conv2d_11_statefulpartitionedcall_args_16
2separable_conv2d_11_statefulpartitionedcall_args_26
2separable_conv2d_11_statefulpartitionedcall_args_36
2separable_conv2d_12_statefulpartitionedcall_args_16
2separable_conv2d_12_statefulpartitionedcall_args_26
2separable_conv2d_12_statefulpartitionedcall_args_36
2separable_conv2d_13_statefulpartitionedcall_args_16
2separable_conv2d_13_statefulpartitionedcall_args_26
2separable_conv2d_13_statefulpartitionedcall_args_36
2separable_conv2d_14_statefulpartitionedcall_args_16
2separable_conv2d_14_statefulpartitionedcall_args_26
2separable_conv2d_14_statefulpartitionedcall_args_36
2separable_conv2d_15_statefulpartitionedcall_args_16
2separable_conv2d_15_statefulpartitionedcall_args_26
2separable_conv2d_15_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв%normalization/StatefulPartitionedCallв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв+separable_conv2d_10/StatefulPartitionedCallв+separable_conv2d_11/StatefulPartitionedCallв+separable_conv2d_12/StatefulPartitionedCallв+separable_conv2d_13/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallв*separable_conv2d_4/StatefulPartitionedCallв*separable_conv2d_5/StatefulPartitionedCallв*separable_conv2d_6/StatefulPartitionedCallв*separable_conv2d_7/StatefulPartitionedCallв*separable_conv2d_8/StatefulPartitionedCallв*separable_conv2d_9/StatefulPartitionedCall═
%normalization/StatefulPartitionedCallStatefulPartitionedCallinputs,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_normalization_layer_call_and_return_conditional_losses_800532'
%normalization/StatefulPartitionedCall╥
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_795452 
conv2d/StatefulPartitionedCall░
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_795702*
(separable_conv2d/StatefulPartitionedCall╞
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_795962,
*separable_conv2d_1/StatefulPartitionedCall╓
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_796172"
 conv2d_1/StatefulPartitionedCallУ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_800852
add/PartitionedCall▒
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_796422,
*separable_conv2d_2/StatefulPartitionedCall╚
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_796682,
*separable_conv2d_3/StatefulPartitionedCallМ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_801082
add_1/PartitionedCall│
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_796942,
*separable_conv2d_4/StatefulPartitionedCall╚
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_797202,
*separable_conv2d_5/StatefulPartitionedCallО
add_2/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_801312
add_2/PartitionedCall│
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_797462,
*separable_conv2d_6/StatefulPartitionedCall╚
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_797722,
*separable_conv2d_7/StatefulPartitionedCallО
add_3/PartitionedCallPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_801542
add_3/PartitionedCall│
*separable_conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:01separable_conv2d_8_statefulpartitionedcall_args_11separable_conv2d_8_statefulpartitionedcall_args_21separable_conv2d_8_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_797982,
*separable_conv2d_8/StatefulPartitionedCall╚
*separable_conv2d_9/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_8/StatefulPartitionedCall:output:01separable_conv2d_9_statefulpartitionedcall_args_11separable_conv2d_9_statefulpartitionedcall_args_21separable_conv2d_9_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_798242,
*separable_conv2d_9/StatefulPartitionedCallО
add_4/PartitionedCallPartitionedCall3separable_conv2d_9/StatefulPartitionedCall:output:0add_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_801772
add_4/PartitionedCall╣
+separable_conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:02separable_conv2d_10_statefulpartitionedcall_args_12separable_conv2d_10_statefulpartitionedcall_args_22separable_conv2d_10_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_798502-
+separable_conv2d_10/StatefulPartitionedCall╧
+separable_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_10/StatefulPartitionedCall:output:02separable_conv2d_11_statefulpartitionedcall_args_12separable_conv2d_11_statefulpartitionedcall_args_22separable_conv2d_11_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_798762-
+separable_conv2d_11/StatefulPartitionedCallП
add_5/PartitionedCallPartitionedCall4separable_conv2d_11/StatefulPartitionedCall:output:0add_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_802002
add_5/PartitionedCall╣
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:02separable_conv2d_12_statefulpartitionedcall_args_12separable_conv2d_12_statefulpartitionedcall_args_22separable_conv2d_12_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_799022-
+separable_conv2d_12/StatefulPartitionedCall╧
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:02separable_conv2d_13_statefulpartitionedcall_args_12separable_conv2d_13_statefulpartitionedcall_args_22separable_conv2d_13_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_799282-
+separable_conv2d_13/StatefulPartitionedCallП
add_6/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0add_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_6_layer_call_and_return_conditional_losses_802232
add_6/PartitionedCall╣
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:02separable_conv2d_14_statefulpartitionedcall_args_12separable_conv2d_14_statefulpartitionedcall_args_22separable_conv2d_14_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_799542-
+separable_conv2d_14/StatefulPartitionedCall╧
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:02separable_conv2d_15_statefulpartitionedcall_args_12separable_conv2d_15_statefulpartitionedcall_args_22separable_conv2d_15_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_799802-
+separable_conv2d_15/StatefulPartitionedCallЖ
max_pooling2d/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_799952
max_pooling2d/PartitionedCall═
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_800132"
 conv2d_2/StatefulPartitionedCallМ
add_7/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_7_layer_call_and_return_conditional_losses_802502
add_7/PartitionedCallЙ
(global_average_pooling2d/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_800282*
(global_average_pooling2d/PartitionedCallД
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_regression_head_1_layer_call_and_return_conditional_losses_802702+
)regression_head_1/StatefulPartitionedCallХ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall,^separable_conv2d_10/StatefulPartitionedCall,^separable_conv2d_11/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall+^separable_conv2d_8/StatefulPartitionedCall+^separable_conv2d_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2Z
+separable_conv2d_10/StatefulPartitionedCall+separable_conv2d_10/StatefulPartitionedCall2Z
+separable_conv2d_11/StatefulPartitionedCall+separable_conv2d_11/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2X
*separable_conv2d_8/StatefulPartitionedCall*separable_conv2d_8/StatefulPartitionedCall2X
*separable_conv2d_9/StatefulPartitionedCall*separable_conv2d_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
б
╫
2__inference_separable_conv2d_6_layer_call_fn_79755

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_797462
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
й
T
8__inference_global_average_pooling2d_layer_call_fn_80034

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:                  *-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_800282
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╝
╧
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_79876

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
°
l
@__inference_add_4_layer_call_and_return_conditional_losses_81441
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╝
╧
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_79980

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ю
h
>__inference_add_layer_call_and_return_conditional_losses_80085

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
╬
Q
%__inference_add_2_layer_call_fn_81423
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_801312
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╗
╬
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_79642

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
▀│
е 
@__inference_model_layer_call_and_return_conditional_losses_80283
input_10
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_35
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_35
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_35
1separable_conv2d_8_statefulpartitionedcall_args_15
1separable_conv2d_8_statefulpartitionedcall_args_25
1separable_conv2d_8_statefulpartitionedcall_args_35
1separable_conv2d_9_statefulpartitionedcall_args_15
1separable_conv2d_9_statefulpartitionedcall_args_25
1separable_conv2d_9_statefulpartitionedcall_args_36
2separable_conv2d_10_statefulpartitionedcall_args_16
2separable_conv2d_10_statefulpartitionedcall_args_26
2separable_conv2d_10_statefulpartitionedcall_args_36
2separable_conv2d_11_statefulpartitionedcall_args_16
2separable_conv2d_11_statefulpartitionedcall_args_26
2separable_conv2d_11_statefulpartitionedcall_args_36
2separable_conv2d_12_statefulpartitionedcall_args_16
2separable_conv2d_12_statefulpartitionedcall_args_26
2separable_conv2d_12_statefulpartitionedcall_args_36
2separable_conv2d_13_statefulpartitionedcall_args_16
2separable_conv2d_13_statefulpartitionedcall_args_26
2separable_conv2d_13_statefulpartitionedcall_args_36
2separable_conv2d_14_statefulpartitionedcall_args_16
2separable_conv2d_14_statefulpartitionedcall_args_26
2separable_conv2d_14_statefulpartitionedcall_args_36
2separable_conv2d_15_statefulpartitionedcall_args_16
2separable_conv2d_15_statefulpartitionedcall_args_26
2separable_conv2d_15_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв%normalization/StatefulPartitionedCallв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв+separable_conv2d_10/StatefulPartitionedCallв+separable_conv2d_11/StatefulPartitionedCallв+separable_conv2d_12/StatefulPartitionedCallв+separable_conv2d_13/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallв*separable_conv2d_4/StatefulPartitionedCallв*separable_conv2d_5/StatefulPartitionedCallв*separable_conv2d_6/StatefulPartitionedCallв*separable_conv2d_7/StatefulPartitionedCallв*separable_conv2d_8/StatefulPartitionedCallв*separable_conv2d_9/StatefulPartitionedCall╬
%normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_normalization_layer_call_and_return_conditional_losses_800532'
%normalization/StatefulPartitionedCall╥
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_795452 
conv2d/StatefulPartitionedCall░
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_795702*
(separable_conv2d/StatefulPartitionedCall╞
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_795962,
*separable_conv2d_1/StatefulPartitionedCall╓
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_796172"
 conv2d_1/StatefulPartitionedCallУ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_800852
add/PartitionedCall▒
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_796422,
*separable_conv2d_2/StatefulPartitionedCall╚
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_796682,
*separable_conv2d_3/StatefulPartitionedCallМ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_801082
add_1/PartitionedCall│
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_796942,
*separable_conv2d_4/StatefulPartitionedCall╚
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_797202,
*separable_conv2d_5/StatefulPartitionedCallО
add_2/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_801312
add_2/PartitionedCall│
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_797462,
*separable_conv2d_6/StatefulPartitionedCall╚
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_797722,
*separable_conv2d_7/StatefulPartitionedCallО
add_3/PartitionedCallPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_801542
add_3/PartitionedCall│
*separable_conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:01separable_conv2d_8_statefulpartitionedcall_args_11separable_conv2d_8_statefulpartitionedcall_args_21separable_conv2d_8_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_797982,
*separable_conv2d_8/StatefulPartitionedCall╚
*separable_conv2d_9/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_8/StatefulPartitionedCall:output:01separable_conv2d_9_statefulpartitionedcall_args_11separable_conv2d_9_statefulpartitionedcall_args_21separable_conv2d_9_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_798242,
*separable_conv2d_9/StatefulPartitionedCallО
add_4/PartitionedCallPartitionedCall3separable_conv2d_9/StatefulPartitionedCall:output:0add_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_801772
add_4/PartitionedCall╣
+separable_conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:02separable_conv2d_10_statefulpartitionedcall_args_12separable_conv2d_10_statefulpartitionedcall_args_22separable_conv2d_10_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_798502-
+separable_conv2d_10/StatefulPartitionedCall╧
+separable_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_10/StatefulPartitionedCall:output:02separable_conv2d_11_statefulpartitionedcall_args_12separable_conv2d_11_statefulpartitionedcall_args_22separable_conv2d_11_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_798762-
+separable_conv2d_11/StatefulPartitionedCallП
add_5/PartitionedCallPartitionedCall4separable_conv2d_11/StatefulPartitionedCall:output:0add_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_802002
add_5/PartitionedCall╣
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:02separable_conv2d_12_statefulpartitionedcall_args_12separable_conv2d_12_statefulpartitionedcall_args_22separable_conv2d_12_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_799022-
+separable_conv2d_12/StatefulPartitionedCall╧
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:02separable_conv2d_13_statefulpartitionedcall_args_12separable_conv2d_13_statefulpartitionedcall_args_22separable_conv2d_13_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_799282-
+separable_conv2d_13/StatefulPartitionedCallП
add_6/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0add_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_6_layer_call_and_return_conditional_losses_802232
add_6/PartitionedCall╣
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:02separable_conv2d_14_statefulpartitionedcall_args_12separable_conv2d_14_statefulpartitionedcall_args_22separable_conv2d_14_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_799542-
+separable_conv2d_14/StatefulPartitionedCall╧
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:02separable_conv2d_15_statefulpartitionedcall_args_12separable_conv2d_15_statefulpartitionedcall_args_22separable_conv2d_15_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_799802-
+separable_conv2d_15/StatefulPartitionedCallЖ
max_pooling2d/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_799952
max_pooling2d/PartitionedCall═
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_800132"
 conv2d_2/StatefulPartitionedCallМ
add_7/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_7_layer_call_and_return_conditional_losses_802502
add_7/PartitionedCallЙ
(global_average_pooling2d/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_800282*
(global_average_pooling2d/PartitionedCallД
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_regression_head_1_layer_call_and_return_conditional_losses_802702+
)regression_head_1/StatefulPartitionedCallХ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall,^separable_conv2d_10/StatefulPartitionedCall,^separable_conv2d_11/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall+^separable_conv2d_8/StatefulPartitionedCall+^separable_conv2d_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2Z
+separable_conv2d_10/StatefulPartitionedCall+separable_conv2d_10/StatefulPartitionedCall2Z
+separable_conv2d_11/StatefulPartitionedCall+separable_conv2d_11/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2X
*separable_conv2d_8/StatefulPartitionedCall*separable_conv2d_8/StatefulPartitionedCall2X
*separable_conv2d_9/StatefulPartitionedCall*separable_conv2d_9/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
╥'
╖
%__inference_model_layer_call_fn_81302

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54#
statefulpartitionedcall_args_55#
statefulpartitionedcall_args_56#
statefulpartitionedcall_args_57#
statefulpartitionedcall_args_58
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54statefulpartitionedcall_args_55statefulpartitionedcall_args_56statefulpartitionedcall_args_57statefulpartitionedcall_args_58*F
Tin?
=2;*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_804722
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
°
l
@__inference_add_3_layer_call_and_return_conditional_losses_81429
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
°
l
@__inference_add_2_layer_call_and_return_conditional_losses_81417
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
▄│
д 
@__inference_model_layer_call_and_return_conditional_losses_80628

inputs0
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_35
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_35
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_35
1separable_conv2d_8_statefulpartitionedcall_args_15
1separable_conv2d_8_statefulpartitionedcall_args_25
1separable_conv2d_8_statefulpartitionedcall_args_35
1separable_conv2d_9_statefulpartitionedcall_args_15
1separable_conv2d_9_statefulpartitionedcall_args_25
1separable_conv2d_9_statefulpartitionedcall_args_36
2separable_conv2d_10_statefulpartitionedcall_args_16
2separable_conv2d_10_statefulpartitionedcall_args_26
2separable_conv2d_10_statefulpartitionedcall_args_36
2separable_conv2d_11_statefulpartitionedcall_args_16
2separable_conv2d_11_statefulpartitionedcall_args_26
2separable_conv2d_11_statefulpartitionedcall_args_36
2separable_conv2d_12_statefulpartitionedcall_args_16
2separable_conv2d_12_statefulpartitionedcall_args_26
2separable_conv2d_12_statefulpartitionedcall_args_36
2separable_conv2d_13_statefulpartitionedcall_args_16
2separable_conv2d_13_statefulpartitionedcall_args_26
2separable_conv2d_13_statefulpartitionedcall_args_36
2separable_conv2d_14_statefulpartitionedcall_args_16
2separable_conv2d_14_statefulpartitionedcall_args_26
2separable_conv2d_14_statefulpartitionedcall_args_36
2separable_conv2d_15_statefulpartitionedcall_args_16
2separable_conv2d_15_statefulpartitionedcall_args_26
2separable_conv2d_15_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв%normalization/StatefulPartitionedCallв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв+separable_conv2d_10/StatefulPartitionedCallв+separable_conv2d_11/StatefulPartitionedCallв+separable_conv2d_12/StatefulPartitionedCallв+separable_conv2d_13/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallв*separable_conv2d_4/StatefulPartitionedCallв*separable_conv2d_5/StatefulPartitionedCallв*separable_conv2d_6/StatefulPartitionedCallв*separable_conv2d_7/StatefulPartitionedCallв*separable_conv2d_8/StatefulPartitionedCallв*separable_conv2d_9/StatefulPartitionedCall═
%normalization/StatefulPartitionedCallStatefulPartitionedCallinputs,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_normalization_layer_call_and_return_conditional_losses_800532'
%normalization/StatefulPartitionedCall╥
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_795452 
conv2d/StatefulPartitionedCall░
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_795702*
(separable_conv2d/StatefulPartitionedCall╞
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_795962,
*separable_conv2d_1/StatefulPartitionedCall╓
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_796172"
 conv2d_1/StatefulPartitionedCallУ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_800852
add/PartitionedCall▒
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_796422,
*separable_conv2d_2/StatefulPartitionedCall╚
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_796682,
*separable_conv2d_3/StatefulPartitionedCallМ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_801082
add_1/PartitionedCall│
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_796942,
*separable_conv2d_4/StatefulPartitionedCall╚
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_797202,
*separable_conv2d_5/StatefulPartitionedCallО
add_2/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_801312
add_2/PartitionedCall│
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_797462,
*separable_conv2d_6/StatefulPartitionedCall╚
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_797722,
*separable_conv2d_7/StatefulPartitionedCallО
add_3/PartitionedCallPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_801542
add_3/PartitionedCall│
*separable_conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:01separable_conv2d_8_statefulpartitionedcall_args_11separable_conv2d_8_statefulpartitionedcall_args_21separable_conv2d_8_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_797982,
*separable_conv2d_8/StatefulPartitionedCall╚
*separable_conv2d_9/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_8/StatefulPartitionedCall:output:01separable_conv2d_9_statefulpartitionedcall_args_11separable_conv2d_9_statefulpartitionedcall_args_21separable_conv2d_9_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_798242,
*separable_conv2d_9/StatefulPartitionedCallО
add_4/PartitionedCallPartitionedCall3separable_conv2d_9/StatefulPartitionedCall:output:0add_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_801772
add_4/PartitionedCall╣
+separable_conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:02separable_conv2d_10_statefulpartitionedcall_args_12separable_conv2d_10_statefulpartitionedcall_args_22separable_conv2d_10_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_798502-
+separable_conv2d_10/StatefulPartitionedCall╧
+separable_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_10/StatefulPartitionedCall:output:02separable_conv2d_11_statefulpartitionedcall_args_12separable_conv2d_11_statefulpartitionedcall_args_22separable_conv2d_11_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_798762-
+separable_conv2d_11/StatefulPartitionedCallП
add_5/PartitionedCallPartitionedCall4separable_conv2d_11/StatefulPartitionedCall:output:0add_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_802002
add_5/PartitionedCall╣
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:02separable_conv2d_12_statefulpartitionedcall_args_12separable_conv2d_12_statefulpartitionedcall_args_22separable_conv2d_12_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_799022-
+separable_conv2d_12/StatefulPartitionedCall╧
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:02separable_conv2d_13_statefulpartitionedcall_args_12separable_conv2d_13_statefulpartitionedcall_args_22separable_conv2d_13_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_799282-
+separable_conv2d_13/StatefulPartitionedCallП
add_6/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0add_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_6_layer_call_and_return_conditional_losses_802232
add_6/PartitionedCall╣
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:02separable_conv2d_14_statefulpartitionedcall_args_12separable_conv2d_14_statefulpartitionedcall_args_22separable_conv2d_14_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_799542-
+separable_conv2d_14/StatefulPartitionedCall╧
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:02separable_conv2d_15_statefulpartitionedcall_args_12separable_conv2d_15_statefulpartitionedcall_args_22separable_conv2d_15_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_799802-
+separable_conv2d_15/StatefulPartitionedCallЖ
max_pooling2d/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_799952
max_pooling2d/PartitionedCall═
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_800132"
 conv2d_2/StatefulPartitionedCallМ
add_7/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_7_layer_call_and_return_conditional_losses_802502
add_7/PartitionedCallЙ
(global_average_pooling2d/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_800282*
(global_average_pooling2d/PartitionedCallД
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_regression_head_1_layer_call_and_return_conditional_losses_802702+
)regression_head_1/StatefulPartitionedCallХ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall,^separable_conv2d_10/StatefulPartitionedCall,^separable_conv2d_11/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall+^separable_conv2d_8/StatefulPartitionedCall+^separable_conv2d_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2Z
+separable_conv2d_10/StatefulPartitionedCall+separable_conv2d_10/StatefulPartitionedCall2Z
+separable_conv2d_11/StatefulPartitionedCall+separable_conv2d_11/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2X
*separable_conv2d_8/StatefulPartitionedCall*separable_conv2d_8/StatefulPartitionedCall2X
*separable_conv2d_9/StatefulPartitionedCall*separable_conv2d_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
╝
╧
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_79850

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
╝
╧
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_79954

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ў

▄
C__inference_conv2d_1_layer_call_and_return_conditional_losses_79617

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_79720

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ё
j
@__inference_add_1_layer_call_and_return_conditional_losses_80108

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
Ё
j
@__inference_add_2_layer_call_and_return_conditional_losses_80131

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_79668

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
б
╫
2__inference_separable_conv2d_1_layer_call_fn_79605

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_795962
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
б
╫
2__inference_separable_conv2d_9_layer_call_fn_79833

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_798242
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
▀│
е 
@__inference_model_layer_call_and_return_conditional_losses_80376
input_10
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_35
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_35
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_35
1separable_conv2d_8_statefulpartitionedcall_args_15
1separable_conv2d_8_statefulpartitionedcall_args_25
1separable_conv2d_8_statefulpartitionedcall_args_35
1separable_conv2d_9_statefulpartitionedcall_args_15
1separable_conv2d_9_statefulpartitionedcall_args_25
1separable_conv2d_9_statefulpartitionedcall_args_36
2separable_conv2d_10_statefulpartitionedcall_args_16
2separable_conv2d_10_statefulpartitionedcall_args_26
2separable_conv2d_10_statefulpartitionedcall_args_36
2separable_conv2d_11_statefulpartitionedcall_args_16
2separable_conv2d_11_statefulpartitionedcall_args_26
2separable_conv2d_11_statefulpartitionedcall_args_36
2separable_conv2d_12_statefulpartitionedcall_args_16
2separable_conv2d_12_statefulpartitionedcall_args_26
2separable_conv2d_12_statefulpartitionedcall_args_36
2separable_conv2d_13_statefulpartitionedcall_args_16
2separable_conv2d_13_statefulpartitionedcall_args_26
2separable_conv2d_13_statefulpartitionedcall_args_36
2separable_conv2d_14_statefulpartitionedcall_args_16
2separable_conv2d_14_statefulpartitionedcall_args_26
2separable_conv2d_14_statefulpartitionedcall_args_36
2separable_conv2d_15_statefulpartitionedcall_args_16
2separable_conv2d_15_statefulpartitionedcall_args_26
2separable_conv2d_15_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв%normalization/StatefulPartitionedCallв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв+separable_conv2d_10/StatefulPartitionedCallв+separable_conv2d_11/StatefulPartitionedCallв+separable_conv2d_12/StatefulPartitionedCallв+separable_conv2d_13/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallв*separable_conv2d_4/StatefulPartitionedCallв*separable_conv2d_5/StatefulPartitionedCallв*separable_conv2d_6/StatefulPartitionedCallв*separable_conv2d_7/StatefulPartitionedCallв*separable_conv2d_8/StatefulPartitionedCallв*separable_conv2d_9/StatefulPartitionedCall╬
%normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_normalization_layer_call_and_return_conditional_losses_800532'
%normalization/StatefulPartitionedCall╥
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_795452 
conv2d/StatefulPartitionedCall░
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_795702*
(separable_conv2d/StatefulPartitionedCall╞
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_795962,
*separable_conv2d_1/StatefulPartitionedCall╓
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_796172"
 conv2d_1/StatefulPartitionedCallУ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_800852
add/PartitionedCall▒
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_796422,
*separable_conv2d_2/StatefulPartitionedCall╚
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_796682,
*separable_conv2d_3/StatefulPartitionedCallМ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_801082
add_1/PartitionedCall│
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_796942,
*separable_conv2d_4/StatefulPartitionedCall╚
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_797202,
*separable_conv2d_5/StatefulPartitionedCallО
add_2/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_801312
add_2/PartitionedCall│
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_797462,
*separable_conv2d_6/StatefulPartitionedCall╚
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_797722,
*separable_conv2d_7/StatefulPartitionedCallО
add_3/PartitionedCallPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_801542
add_3/PartitionedCall│
*separable_conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:01separable_conv2d_8_statefulpartitionedcall_args_11separable_conv2d_8_statefulpartitionedcall_args_21separable_conv2d_8_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_797982,
*separable_conv2d_8/StatefulPartitionedCall╚
*separable_conv2d_9/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_8/StatefulPartitionedCall:output:01separable_conv2d_9_statefulpartitionedcall_args_11separable_conv2d_9_statefulpartitionedcall_args_21separable_conv2d_9_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_798242,
*separable_conv2d_9/StatefulPartitionedCallО
add_4/PartitionedCallPartitionedCall3separable_conv2d_9/StatefulPartitionedCall:output:0add_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_801772
add_4/PartitionedCall╣
+separable_conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:02separable_conv2d_10_statefulpartitionedcall_args_12separable_conv2d_10_statefulpartitionedcall_args_22separable_conv2d_10_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_798502-
+separable_conv2d_10/StatefulPartitionedCall╧
+separable_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_10/StatefulPartitionedCall:output:02separable_conv2d_11_statefulpartitionedcall_args_12separable_conv2d_11_statefulpartitionedcall_args_22separable_conv2d_11_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_798762-
+separable_conv2d_11/StatefulPartitionedCallП
add_5/PartitionedCallPartitionedCall4separable_conv2d_11/StatefulPartitionedCall:output:0add_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_5_layer_call_and_return_conditional_losses_802002
add_5/PartitionedCall╣
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:02separable_conv2d_12_statefulpartitionedcall_args_12separable_conv2d_12_statefulpartitionedcall_args_22separable_conv2d_12_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_799022-
+separable_conv2d_12/StatefulPartitionedCall╧
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:02separable_conv2d_13_statefulpartitionedcall_args_12separable_conv2d_13_statefulpartitionedcall_args_22separable_conv2d_13_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_799282-
+separable_conv2d_13/StatefulPartitionedCallП
add_6/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0add_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_6_layer_call_and_return_conditional_losses_802232
add_6/PartitionedCall╣
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:02separable_conv2d_14_statefulpartitionedcall_args_12separable_conv2d_14_statefulpartitionedcall_args_22separable_conv2d_14_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_799542-
+separable_conv2d_14/StatefulPartitionedCall╧
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:02separable_conv2d_15_statefulpartitionedcall_args_12separable_conv2d_15_statefulpartitionedcall_args_22separable_conv2d_15_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_799802-
+separable_conv2d_15/StatefulPartitionedCallЖ
max_pooling2d/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_799952
max_pooling2d/PartitionedCall═
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_800132"
 conv2d_2/StatefulPartitionedCallМ
add_7/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_7_layer_call_and_return_conditional_losses_802502
add_7/PartitionedCallЙ
(global_average_pooling2d/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_800282*
(global_average_pooling2d/PartitionedCallД
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_regression_head_1_layer_call_and_return_conditional_losses_802702+
)regression_head_1/StatefulPartitionedCallХ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall,^separable_conv2d_10/StatefulPartitionedCall,^separable_conv2d_11/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall+^separable_conv2d_8/StatefulPartitionedCall+^separable_conv2d_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2Z
+separable_conv2d_10/StatefulPartitionedCall+separable_conv2d_10/StatefulPartitionedCall2Z
+separable_conv2d_11/StatefulPartitionedCall+separable_conv2d_11/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2X
*separable_conv2d_8/StatefulPartitionedCall*separable_conv2d_8/StatefulPartitionedCall2X
*separable_conv2d_9/StatefulPartitionedCall*separable_conv2d_9/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
╡к
Г2
@__inference_model_layer_call_and_return_conditional_losses_80996

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource=
9separable_conv2d_separable_conv2d_readvariableop_resource?
;separable_conv2d_separable_conv2d_readvariableop_1_resource4
0separable_conv2d_biasadd_readvariableop_resource?
;separable_conv2d_1_separable_conv2d_readvariableop_resourceA
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_1_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource?
;separable_conv2d_2_separable_conv2d_readvariableop_resourceA
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_2_biasadd_readvariableop_resource?
;separable_conv2d_3_separable_conv2d_readvariableop_resourceA
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_3_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource?
;separable_conv2d_6_separable_conv2d_readvariableop_resourceA
=separable_conv2d_6_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_6_biasadd_readvariableop_resource?
;separable_conv2d_7_separable_conv2d_readvariableop_resourceA
=separable_conv2d_7_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_7_biasadd_readvariableop_resource?
;separable_conv2d_8_separable_conv2d_readvariableop_resourceA
=separable_conv2d_8_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_8_biasadd_readvariableop_resource?
;separable_conv2d_9_separable_conv2d_readvariableop_resourceA
=separable_conv2d_9_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_9_biasadd_readvariableop_resource@
<separable_conv2d_10_separable_conv2d_readvariableop_resourceB
>separable_conv2d_10_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_10_biasadd_readvariableop_resource@
<separable_conv2d_11_separable_conv2d_readvariableop_resourceB
>separable_conv2d_11_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_11_biasadd_readvariableop_resource@
<separable_conv2d_12_separable_conv2d_readvariableop_resourceB
>separable_conv2d_12_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_12_biasadd_readvariableop_resource@
<separable_conv2d_13_separable_conv2d_readvariableop_resourceB
>separable_conv2d_13_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_13_biasadd_readvariableop_resource@
<separable_conv2d_14_separable_conv2d_readvariableop_resourceB
>separable_conv2d_14_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_14_biasadd_readvariableop_resource@
<separable_conv2d_15_separable_conv2d_readvariableop_resourceB
>separable_conv2d_15_separable_conv2d_readvariableop_1_resource7
3separable_conv2d_15_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource4
0regression_head_1_matmul_readvariableop_resource5
1regression_head_1_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpв$normalization/Reshape/ReadVariableOpв&normalization/Reshape_1/ReadVariableOpв(regression_head_1/BiasAdd/ReadVariableOpв'regression_head_1/MatMul/ReadVariableOpв'separable_conv2d/BiasAdd/ReadVariableOpв0separable_conv2d/separable_conv2d/ReadVariableOpв2separable_conv2d/separable_conv2d/ReadVariableOp_1в)separable_conv2d_1/BiasAdd/ReadVariableOpв2separable_conv2d_1/separable_conv2d/ReadVariableOpв4separable_conv2d_1/separable_conv2d/ReadVariableOp_1в*separable_conv2d_10/BiasAdd/ReadVariableOpв3separable_conv2d_10/separable_conv2d/ReadVariableOpв5separable_conv2d_10/separable_conv2d/ReadVariableOp_1в*separable_conv2d_11/BiasAdd/ReadVariableOpв3separable_conv2d_11/separable_conv2d/ReadVariableOpв5separable_conv2d_11/separable_conv2d/ReadVariableOp_1в*separable_conv2d_12/BiasAdd/ReadVariableOpв3separable_conv2d_12/separable_conv2d/ReadVariableOpв5separable_conv2d_12/separable_conv2d/ReadVariableOp_1в*separable_conv2d_13/BiasAdd/ReadVariableOpв3separable_conv2d_13/separable_conv2d/ReadVariableOpв5separable_conv2d_13/separable_conv2d/ReadVariableOp_1в*separable_conv2d_14/BiasAdd/ReadVariableOpв3separable_conv2d_14/separable_conv2d/ReadVariableOpв5separable_conv2d_14/separable_conv2d/ReadVariableOp_1в*separable_conv2d_15/BiasAdd/ReadVariableOpв3separable_conv2d_15/separable_conv2d/ReadVariableOpв5separable_conv2d_15/separable_conv2d/ReadVariableOp_1в)separable_conv2d_2/BiasAdd/ReadVariableOpв2separable_conv2d_2/separable_conv2d/ReadVariableOpв4separable_conv2d_2/separable_conv2d/ReadVariableOp_1в)separable_conv2d_3/BiasAdd/ReadVariableOpв2separable_conv2d_3/separable_conv2d/ReadVariableOpв4separable_conv2d_3/separable_conv2d/ReadVariableOp_1в)separable_conv2d_4/BiasAdd/ReadVariableOpв2separable_conv2d_4/separable_conv2d/ReadVariableOpв4separable_conv2d_4/separable_conv2d/ReadVariableOp_1в)separable_conv2d_5/BiasAdd/ReadVariableOpв2separable_conv2d_5/separable_conv2d/ReadVariableOpв4separable_conv2d_5/separable_conv2d/ReadVariableOp_1в)separable_conv2d_6/BiasAdd/ReadVariableOpв2separable_conv2d_6/separable_conv2d/ReadVariableOpв4separable_conv2d_6/separable_conv2d/ReadVariableOp_1в)separable_conv2d_7/BiasAdd/ReadVariableOpв2separable_conv2d_7/separable_conv2d/ReadVariableOpв4separable_conv2d_7/separable_conv2d/ReadVariableOp_1в)separable_conv2d_8/BiasAdd/ReadVariableOpв2separable_conv2d_8/separable_conv2d/ReadVariableOpв4separable_conv2d_8/separable_conv2d/ReadVariableOp_1в)separable_conv2d_9/BiasAdd/ReadVariableOpв2separable_conv2d_9/separable_conv2d/ReadVariableOpв4separable_conv2d_9/separable_conv2d/ReadVariableOp_1╢
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOpУ
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape╛
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape╝
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOpЧ
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape╞
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1П
normalization/subSubinputsnormalization/Reshape:output:0*
T0*/
_output_shapes
:         @@2
normalization/subГ
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtв
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*/
_output_shapes
:         @@2
normalization/truedivк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╦
conv2d/Conv2DConv2Dnormalization/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:           @2
conv2d/Seluц
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpэ
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1л
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2)
'separable_conv2d/separable_conv2d/Shape│
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rateк
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwiseж
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2d└
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp╫
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d/BiasAddФ
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d/Seluэ
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_1/separable_conv2d/Shape╖
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rate╗
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwiseо
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d╞
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOp▀
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_1/BiasAddЪ
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_1/Selu▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╥
conv2d_1/Conv2DConv2Dconv2d/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpн
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
conv2d_1/BiasAddШ
add/addAddV2%separable_conv2d_1/Selu:activations:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:           А2	
add/addэ
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_2/separable_conv2d/Shape╖
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rateг
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativeadd/add:z:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwiseо
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2d╞
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOp▀
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_2/BiasAddЪ
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_2/Seluэ
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_3/separable_conv2d/Shape╖
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rate╜
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_2/Selu:activations:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwiseо
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2d╞
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOp▀
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_3/BiasAddЪ
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_3/SeluО
	add_1/addAddV2%separable_conv2d_3/Selu:activations:0add/add:z:0*
T0*0
_output_shapes
:           А2
	add_1/addэ
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_4/separable_conv2d/Shape╖
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rateе
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_1/add:z:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwiseо
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2d╞
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOp▀
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_4/BiasAddЪ
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_4/Seluэ
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_5/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_5/separable_conv2d/Shape╖
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_5/separable_conv2d/dilation_rate╜
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_4/Selu:activations:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_5/separable_conv2d/depthwiseо
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_5/separable_conv2d╞
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_5/BiasAdd/ReadVariableOp▀
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_5/BiasAddЪ
separable_conv2d_5/SeluSelu#separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_5/SeluР
	add_2/addAddV2%separable_conv2d_5/Selu:activations:0add_1/add:z:0*
T0*0
_output_shapes
:           А2
	add_2/addэ
2separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_6_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_6/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_6_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_6/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_6/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_6/separable_conv2d/Shape╖
1separable_conv2d_6/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_6/separable_conv2d/dilation_rateе
-separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_2/add:z:0:separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_6/separable_conv2d/depthwiseо
#separable_conv2d_6/separable_conv2dConv2D6separable_conv2d_6/separable_conv2d/depthwise:output:0<separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_6/separable_conv2d╞
)separable_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_6/BiasAdd/ReadVariableOp▀
separable_conv2d_6/BiasAddBiasAdd,separable_conv2d_6/separable_conv2d:output:01separable_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_6/BiasAddЪ
separable_conv2d_6/SeluSelu#separable_conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_6/Seluэ
2separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_7_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_7/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_7_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_7/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_7/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_7/separable_conv2d/Shape╖
1separable_conv2d_7/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_7/separable_conv2d/dilation_rate╜
-separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_6/Selu:activations:0:separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_7/separable_conv2d/depthwiseо
#separable_conv2d_7/separable_conv2dConv2D6separable_conv2d_7/separable_conv2d/depthwise:output:0<separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_7/separable_conv2d╞
)separable_conv2d_7/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_7/BiasAdd/ReadVariableOp▀
separable_conv2d_7/BiasAddBiasAdd,separable_conv2d_7/separable_conv2d:output:01separable_conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_7/BiasAddЪ
separable_conv2d_7/SeluSelu#separable_conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_7/SeluР
	add_3/addAddV2%separable_conv2d_7/Selu:activations:0add_2/add:z:0*
T0*0
_output_shapes
:           А2
	add_3/addэ
2separable_conv2d_8/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_8_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_8/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_8/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_8_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_8/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_8/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_8/separable_conv2d/Shape╖
1separable_conv2d_8/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_8/separable_conv2d/dilation_rateе
-separable_conv2d_8/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_3/add:z:0:separable_conv2d_8/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_8/separable_conv2d/depthwiseо
#separable_conv2d_8/separable_conv2dConv2D6separable_conv2d_8/separable_conv2d/depthwise:output:0<separable_conv2d_8/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_8/separable_conv2d╞
)separable_conv2d_8/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_8/BiasAdd/ReadVariableOp▀
separable_conv2d_8/BiasAddBiasAdd,separable_conv2d_8/separable_conv2d:output:01separable_conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_8/BiasAddЪ
separable_conv2d_8/SeluSelu#separable_conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_8/Seluэ
2separable_conv2d_9/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_9_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_9/separable_conv2d/ReadVariableOpЇ
4separable_conv2d_9/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_9_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_9/separable_conv2d/ReadVariableOp_1п
)separable_conv2d_9/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_9/separable_conv2d/Shape╖
1separable_conv2d_9/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_9/separable_conv2d/dilation_rate╜
-separable_conv2d_9/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_8/Selu:activations:0:separable_conv2d_9/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2/
-separable_conv2d_9/separable_conv2d/depthwiseо
#separable_conv2d_9/separable_conv2dConv2D6separable_conv2d_9/separable_conv2d/depthwise:output:0<separable_conv2d_9/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2%
#separable_conv2d_9/separable_conv2d╞
)separable_conv2d_9/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_9/BiasAdd/ReadVariableOp▀
separable_conv2d_9/BiasAddBiasAdd,separable_conv2d_9/separable_conv2d:output:01separable_conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_9/BiasAddЪ
separable_conv2d_9/SeluSelu#separable_conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_9/SeluР
	add_4/addAddV2%separable_conv2d_9/Selu:activations:0add_3/add:z:0*
T0*0
_output_shapes
:           А2
	add_4/addЁ
3separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_10_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_10/separable_conv2d/ReadVariableOpў
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_10_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_10/separable_conv2d/Shape╣
2separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_10/separable_conv2d/dilation_rateи
.separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_4/add:z:0;separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_10/separable_conv2d/depthwise▓
$separable_conv2d_10/separable_conv2dConv2D7separable_conv2d_10/separable_conv2d/depthwise:output:0=separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_10/separable_conv2d╔
*separable_conv2d_10/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_10/BiasAdd/ReadVariableOpу
separable_conv2d_10/BiasAddBiasAdd-separable_conv2d_10/separable_conv2d:output:02separable_conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_10/BiasAddЭ
separable_conv2d_10/SeluSelu$separable_conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_10/SeluЁ
3separable_conv2d_11/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_11_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_11/separable_conv2d/ReadVariableOpў
5separable_conv2d_11/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_11_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_11/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_11/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_11/separable_conv2d/Shape╣
2separable_conv2d_11/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_11/separable_conv2d/dilation_rate┴
.separable_conv2d_11/separable_conv2d/depthwiseDepthwiseConv2dNative&separable_conv2d_10/Selu:activations:0;separable_conv2d_11/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_11/separable_conv2d/depthwise▓
$separable_conv2d_11/separable_conv2dConv2D7separable_conv2d_11/separable_conv2d/depthwise:output:0=separable_conv2d_11/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_11/separable_conv2d╔
*separable_conv2d_11/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_11/BiasAdd/ReadVariableOpу
separable_conv2d_11/BiasAddBiasAdd-separable_conv2d_11/separable_conv2d:output:02separable_conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_11/BiasAddЭ
separable_conv2d_11/SeluSelu$separable_conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_11/SeluС
	add_5/addAddV2&separable_conv2d_11/Selu:activations:0add_4/add:z:0*
T0*0
_output_shapes
:           А2
	add_5/addЁ
3separable_conv2d_12/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_12_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_12/separable_conv2d/ReadVariableOpў
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_12_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_12/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_12/separable_conv2d/Shape╣
2separable_conv2d_12/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_12/separable_conv2d/dilation_rateи
.separable_conv2d_12/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_5/add:z:0;separable_conv2d_12/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_12/separable_conv2d/depthwise▓
$separable_conv2d_12/separable_conv2dConv2D7separable_conv2d_12/separable_conv2d/depthwise:output:0=separable_conv2d_12/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_12/separable_conv2d╔
*separable_conv2d_12/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_12/BiasAdd/ReadVariableOpу
separable_conv2d_12/BiasAddBiasAdd-separable_conv2d_12/separable_conv2d:output:02separable_conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_12/BiasAddЭ
separable_conv2d_12/SeluSelu$separable_conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_12/SeluЁ
3separable_conv2d_13/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_13_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_13/separable_conv2d/ReadVariableOpў
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_13_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_13/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_13/separable_conv2d/Shape╣
2separable_conv2d_13/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_13/separable_conv2d/dilation_rate┴
.separable_conv2d_13/separable_conv2d/depthwiseDepthwiseConv2dNative&separable_conv2d_12/Selu:activations:0;separable_conv2d_13/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_13/separable_conv2d/depthwise▓
$separable_conv2d_13/separable_conv2dConv2D7separable_conv2d_13/separable_conv2d/depthwise:output:0=separable_conv2d_13/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_13/separable_conv2d╔
*separable_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_13/BiasAdd/ReadVariableOpу
separable_conv2d_13/BiasAddBiasAdd-separable_conv2d_13/separable_conv2d:output:02separable_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_13/BiasAddЭ
separable_conv2d_13/SeluSelu$separable_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_13/SeluС
	add_6/addAddV2&separable_conv2d_13/Selu:activations:0add_5/add:z:0*
T0*0
_output_shapes
:           А2
	add_6/addЁ
3separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_14_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_14/separable_conv2d/ReadVariableOpў
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_14_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_14/separable_conv2d/Shape╣
2separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_14/separable_conv2d/dilation_rateи
.separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_6/add:z:0;separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_14/separable_conv2d/depthwise▓
$separable_conv2d_14/separable_conv2dConv2D7separable_conv2d_14/separable_conv2d/depthwise:output:0=separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_14/separable_conv2d╔
*separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_14/BiasAdd/ReadVariableOpу
separable_conv2d_14/BiasAddBiasAdd-separable_conv2d_14/separable_conv2d:output:02separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_14/BiasAddЭ
separable_conv2d_14/SeluSelu$separable_conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_14/SeluЁ
3separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_15_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_15/separable_conv2d/ReadVariableOpў
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_15_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_15/separable_conv2d/Shape╣
2separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_15/separable_conv2d/dilation_rate┴
.separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative&separable_conv2d_14/Selu:activations:0;separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_15/separable_conv2d/depthwise▓
$separable_conv2d_15/separable_conv2dConv2D7separable_conv2d_15/separable_conv2d/depthwise:output:0=separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_15/separable_conv2d╔
*separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_15/BiasAdd/ReadVariableOpу
separable_conv2d_15/BiasAddBiasAdd-separable_conv2d_15/separable_conv2d:output:02separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_15/BiasAddЭ
separable_conv2d_15/SeluSelu$separable_conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
separable_conv2d_15/Selu╬
max_pooling2d/MaxPoolMaxPool&separable_conv2d_15/Selu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool▓
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╞
conv2d_2/Conv2DConv2Dadd_6/add:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_2/BiasAddХ
	add_7/addAddV2max_pooling2d/MaxPool:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
	add_7/add│
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices┬
global_average_pooling2d/MeanMeanadd_7/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
global_average_pooling2d/Mean─
'regression_head_1/MatMul/ReadVariableOpReadVariableOp0regression_head_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'regression_head_1/MatMul/ReadVariableOp╔
regression_head_1/MatMulMatMul&global_average_pooling2d/Mean:output:0/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
regression_head_1/MatMul┬
(regression_head_1/BiasAdd/ReadVariableOpReadVariableOp1regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(regression_head_1/BiasAdd/ReadVariableOp╔
regression_head_1/BiasAddBiasAdd"regression_head_1/MatMul:product:00regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
regression_head_1/BiasAddь
IdentityIdentity"regression_head_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp)^regression_head_1/BiasAdd/ReadVariableOp(^regression_head_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1+^separable_conv2d_10/BiasAdd/ReadVariableOp4^separable_conv2d_10/separable_conv2d/ReadVariableOp6^separable_conv2d_10/separable_conv2d/ReadVariableOp_1+^separable_conv2d_11/BiasAdd/ReadVariableOp4^separable_conv2d_11/separable_conv2d/ReadVariableOp6^separable_conv2d_11/separable_conv2d/ReadVariableOp_1+^separable_conv2d_12/BiasAdd/ReadVariableOp4^separable_conv2d_12/separable_conv2d/ReadVariableOp6^separable_conv2d_12/separable_conv2d/ReadVariableOp_1+^separable_conv2d_13/BiasAdd/ReadVariableOp4^separable_conv2d_13/separable_conv2d/ReadVariableOp6^separable_conv2d_13/separable_conv2d/ReadVariableOp_1+^separable_conv2d_14/BiasAdd/ReadVariableOp4^separable_conv2d_14/separable_conv2d/ReadVariableOp6^separable_conv2d_14/separable_conv2d/ReadVariableOp_1+^separable_conv2d_15/BiasAdd/ReadVariableOp4^separable_conv2d_15/separable_conv2d/ReadVariableOp6^separable_conv2d_15/separable_conv2d/ReadVariableOp_1*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*^separable_conv2d_5/BiasAdd/ReadVariableOp3^separable_conv2d_5/separable_conv2d/ReadVariableOp5^separable_conv2d_5/separable_conv2d/ReadVariableOp_1*^separable_conv2d_6/BiasAdd/ReadVariableOp3^separable_conv2d_6/separable_conv2d/ReadVariableOp5^separable_conv2d_6/separable_conv2d/ReadVariableOp_1*^separable_conv2d_7/BiasAdd/ReadVariableOp3^separable_conv2d_7/separable_conv2d/ReadVariableOp5^separable_conv2d_7/separable_conv2d/ReadVariableOp_1*^separable_conv2d_8/BiasAdd/ReadVariableOp3^separable_conv2d_8/separable_conv2d/ReadVariableOp5^separable_conv2d_8/separable_conv2d/ReadVariableOp_1*^separable_conv2d_9/BiasAdd/ReadVariableOp3^separable_conv2d_9/separable_conv2d/ReadVariableOp5^separable_conv2d_9/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2T
(regression_head_1/BiasAdd/ReadVariableOp(regression_head_1/BiasAdd/ReadVariableOp2R
'regression_head_1/MatMul/ReadVariableOp'regression_head_1/MatMul/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_10/BiasAdd/ReadVariableOp*separable_conv2d_10/BiasAdd/ReadVariableOp2j
3separable_conv2d_10/separable_conv2d/ReadVariableOp3separable_conv2d_10/separable_conv2d/ReadVariableOp2n
5separable_conv2d_10/separable_conv2d/ReadVariableOp_15separable_conv2d_10/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_11/BiasAdd/ReadVariableOp*separable_conv2d_11/BiasAdd/ReadVariableOp2j
3separable_conv2d_11/separable_conv2d/ReadVariableOp3separable_conv2d_11/separable_conv2d/ReadVariableOp2n
5separable_conv2d_11/separable_conv2d/ReadVariableOp_15separable_conv2d_11/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_12/BiasAdd/ReadVariableOp*separable_conv2d_12/BiasAdd/ReadVariableOp2j
3separable_conv2d_12/separable_conv2d/ReadVariableOp3separable_conv2d_12/separable_conv2d/ReadVariableOp2n
5separable_conv2d_12/separable_conv2d/ReadVariableOp_15separable_conv2d_12/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_13/BiasAdd/ReadVariableOp*separable_conv2d_13/BiasAdd/ReadVariableOp2j
3separable_conv2d_13/separable_conv2d/ReadVariableOp3separable_conv2d_13/separable_conv2d/ReadVariableOp2n
5separable_conv2d_13/separable_conv2d/ReadVariableOp_15separable_conv2d_13/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_14/BiasAdd/ReadVariableOp*separable_conv2d_14/BiasAdd/ReadVariableOp2j
3separable_conv2d_14/separable_conv2d/ReadVariableOp3separable_conv2d_14/separable_conv2d/ReadVariableOp2n
5separable_conv2d_14/separable_conv2d/ReadVariableOp_15separable_conv2d_14/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_15/BiasAdd/ReadVariableOp*separable_conv2d_15/BiasAdd/ReadVariableOp2j
3separable_conv2d_15/separable_conv2d/ReadVariableOp3separable_conv2d_15/separable_conv2d/ReadVariableOp2n
5separable_conv2d_15/separable_conv2d/ReadVariableOp_15separable_conv2d_15/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_5/BiasAdd/ReadVariableOp)separable_conv2d_5/BiasAdd/ReadVariableOp2h
2separable_conv2d_5/separable_conv2d/ReadVariableOp2separable_conv2d_5/separable_conv2d/ReadVariableOp2l
4separable_conv2d_5/separable_conv2d/ReadVariableOp_14separable_conv2d_5/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_6/BiasAdd/ReadVariableOp)separable_conv2d_6/BiasAdd/ReadVariableOp2h
2separable_conv2d_6/separable_conv2d/ReadVariableOp2separable_conv2d_6/separable_conv2d/ReadVariableOp2l
4separable_conv2d_6/separable_conv2d/ReadVariableOp_14separable_conv2d_6/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_7/BiasAdd/ReadVariableOp)separable_conv2d_7/BiasAdd/ReadVariableOp2h
2separable_conv2d_7/separable_conv2d/ReadVariableOp2separable_conv2d_7/separable_conv2d/ReadVariableOp2l
4separable_conv2d_7/separable_conv2d/ReadVariableOp_14separable_conv2d_7/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_8/BiasAdd/ReadVariableOp)separable_conv2d_8/BiasAdd/ReadVariableOp2h
2separable_conv2d_8/separable_conv2d/ReadVariableOp2separable_conv2d_8/separable_conv2d/ReadVariableOp2l
4separable_conv2d_8/separable_conv2d/ReadVariableOp_14separable_conv2d_8/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_9/BiasAdd/ReadVariableOp)separable_conv2d_9/BiasAdd/ReadVariableOp2h
2separable_conv2d_9/separable_conv2d/ReadVariableOp2separable_conv2d_9/separable_conv2d/ReadVariableOp2l
4separable_conv2d_9/separable_conv2d/ReadVariableOp_14separable_conv2d_9/separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
°
l
@__inference_add_5_layer_call_and_return_conditional_losses_81453
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╬
Q
%__inference_add_4_layer_call_fn_81447
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_4_layer_call_and_return_conditional_losses_801772
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
├
й
(__inference_conv2d_1_layer_call_fn_79625

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_796172
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
г
╪
3__inference_separable_conv2d_10_layer_call_fn_79859

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_798502
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
б
╫
2__inference_separable_conv2d_8_layer_call_fn_79807

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_797982
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
°
l
@__inference_add_6_layer_call_and_return_conditional_losses_81465
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╚
I
-__inference_max_pooling2d_layer_call_fn_80001

inputs
identity╓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_799952
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╬
Q
%__inference_add_1_layer_call_fn_81411
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_801082
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
б
╫
2__inference_separable_conv2d_3_layer_call_fn_79677

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_796682
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╬
Q
%__inference_add_7_layer_call_fn_81483
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_7_layer_call_and_return_conditional_losses_802502
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:         А:         А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ў
j
>__inference_add_layer_call_and_return_conditional_losses_81393
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╬
Q
%__inference_add_6_layer_call_fn_81471
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_6_layer_call_and_return_conditional_losses_802232
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╝
╧
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_79902

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_79694

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
╗
╬
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_79824

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
╩
O
#__inference_add_layer_call_fn_81399
inputs_0
inputs_1
identity┐
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_800852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
°
l
@__inference_add_7_layer_call_and_return_conditional_losses_81477
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:         А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:         А:         А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╤
ч
H__inference_normalization_layer_call_and_return_conditional_losses_81380

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identityИвReshape/ReadVariableOpвReshape_1/ReadVariableOpМ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shapeЖ
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
ReshapeТ
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shapeО
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1e
subSubinputsReshape:output:0*
T0*/
_output_shapes
:         @@2
subY
SqrtSqrtReshape_1:output:0*
T0*&
_output_shapes
:2
Sqrtj
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:         @@2	
truedivЫ
IdentityIdentitytruediv:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:& "
 
_user_specified_nameinputs
є
х
L__inference_regression_head_1_layer_call_and_return_conditional_losses_81493

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╤
ч
H__inference_normalization_layer_call_and_return_conditional_losses_80053

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identityИвReshape/ReadVariableOpвReshape_1/ReadVariableOpМ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shapeЖ
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
ReshapeТ
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shapeО
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1e
subSubinputsReshape:output:0*
T0*/
_output_shapes
:         @@2
subY
SqrtSqrtReshape_1:output:0*
T0*&
_output_shapes
:2
Sqrtj
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:         @@2	
truedivЫ
IdentityIdentitytruediv:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:& "
 
_user_specified_nameinputs
ы╥
Ь7
 __inference__wrapped_model_79532
input_17
3model_normalization_reshape_readvariableop_resource9
5model_normalization_reshape_1_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resourceC
?model_separable_conv2d_separable_conv2d_readvariableop_resourceE
Amodel_separable_conv2d_separable_conv2d_readvariableop_1_resource:
6model_separable_conv2d_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_1_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_1_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_2_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_2_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_2_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_3_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_3_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_3_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_4_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_4_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_4_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_5_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_5_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_5_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_6_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_6_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_6_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_7_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_7_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_7_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_8_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_8_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_8_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_9_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_9_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_9_biasadd_readvariableop_resourceF
Bmodel_separable_conv2d_10_separable_conv2d_readvariableop_resourceH
Dmodel_separable_conv2d_10_separable_conv2d_readvariableop_1_resource=
9model_separable_conv2d_10_biasadd_readvariableop_resourceF
Bmodel_separable_conv2d_11_separable_conv2d_readvariableop_resourceH
Dmodel_separable_conv2d_11_separable_conv2d_readvariableop_1_resource=
9model_separable_conv2d_11_biasadd_readvariableop_resourceF
Bmodel_separable_conv2d_12_separable_conv2d_readvariableop_resourceH
Dmodel_separable_conv2d_12_separable_conv2d_readvariableop_1_resource=
9model_separable_conv2d_12_biasadd_readvariableop_resourceF
Bmodel_separable_conv2d_13_separable_conv2d_readvariableop_resourceH
Dmodel_separable_conv2d_13_separable_conv2d_readvariableop_1_resource=
9model_separable_conv2d_13_biasadd_readvariableop_resourceF
Bmodel_separable_conv2d_14_separable_conv2d_readvariableop_resourceH
Dmodel_separable_conv2d_14_separable_conv2d_readvariableop_1_resource=
9model_separable_conv2d_14_biasadd_readvariableop_resourceF
Bmodel_separable_conv2d_15_separable_conv2d_readvariableop_resourceH
Dmodel_separable_conv2d_15_separable_conv2d_readvariableop_1_resource=
9model_separable_conv2d_15_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource:
6model_regression_head_1_matmul_readvariableop_resource;
7model_regression_head_1_biasadd_readvariableop_resource
identityИв#model/conv2d/BiasAdd/ReadVariableOpв"model/conv2d/Conv2D/ReadVariableOpв%model/conv2d_1/BiasAdd/ReadVariableOpв$model/conv2d_1/Conv2D/ReadVariableOpв%model/conv2d_2/BiasAdd/ReadVariableOpв$model/conv2d_2/Conv2D/ReadVariableOpв*model/normalization/Reshape/ReadVariableOpв,model/normalization/Reshape_1/ReadVariableOpв.model/regression_head_1/BiasAdd/ReadVariableOpв-model/regression_head_1/MatMul/ReadVariableOpв-model/separable_conv2d/BiasAdd/ReadVariableOpв6model/separable_conv2d/separable_conv2d/ReadVariableOpв8model/separable_conv2d/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_1/BiasAdd/ReadVariableOpв8model/separable_conv2d_1/separable_conv2d/ReadVariableOpв:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1в0model/separable_conv2d_10/BiasAdd/ReadVariableOpв9model/separable_conv2d_10/separable_conv2d/ReadVariableOpв;model/separable_conv2d_10/separable_conv2d/ReadVariableOp_1в0model/separable_conv2d_11/BiasAdd/ReadVariableOpв9model/separable_conv2d_11/separable_conv2d/ReadVariableOpв;model/separable_conv2d_11/separable_conv2d/ReadVariableOp_1в0model/separable_conv2d_12/BiasAdd/ReadVariableOpв9model/separable_conv2d_12/separable_conv2d/ReadVariableOpв;model/separable_conv2d_12/separable_conv2d/ReadVariableOp_1в0model/separable_conv2d_13/BiasAdd/ReadVariableOpв9model/separable_conv2d_13/separable_conv2d/ReadVariableOpв;model/separable_conv2d_13/separable_conv2d/ReadVariableOp_1в0model/separable_conv2d_14/BiasAdd/ReadVariableOpв9model/separable_conv2d_14/separable_conv2d/ReadVariableOpв;model/separable_conv2d_14/separable_conv2d/ReadVariableOp_1в0model/separable_conv2d_15/BiasAdd/ReadVariableOpв9model/separable_conv2d_15/separable_conv2d/ReadVariableOpв;model/separable_conv2d_15/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_2/BiasAdd/ReadVariableOpв8model/separable_conv2d_2/separable_conv2d/ReadVariableOpв:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_3/BiasAdd/ReadVariableOpв8model/separable_conv2d_3/separable_conv2d/ReadVariableOpв:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_4/BiasAdd/ReadVariableOpв8model/separable_conv2d_4/separable_conv2d/ReadVariableOpв:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_5/BiasAdd/ReadVariableOpв8model/separable_conv2d_5/separable_conv2d/ReadVariableOpв:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_6/BiasAdd/ReadVariableOpв8model/separable_conv2d_6/separable_conv2d/ReadVariableOpв:model/separable_conv2d_6/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_7/BiasAdd/ReadVariableOpв8model/separable_conv2d_7/separable_conv2d/ReadVariableOpв:model/separable_conv2d_7/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_8/BiasAdd/ReadVariableOpв8model/separable_conv2d_8/separable_conv2d/ReadVariableOpв:model/separable_conv2d_8/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_9/BiasAdd/ReadVariableOpв8model/separable_conv2d_9/separable_conv2d/ReadVariableOpв:model/separable_conv2d_9/separable_conv2d/ReadVariableOp_1╚
*model/normalization/Reshape/ReadVariableOpReadVariableOp3model_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/normalization/Reshape/ReadVariableOpЯ
!model/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!model/normalization/Reshape/shape╓
model/normalization/ReshapeReshape2model/normalization/Reshape/ReadVariableOp:value:0*model/normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
model/normalization/Reshape╬
,model/normalization/Reshape_1/ReadVariableOpReadVariableOp5model_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,model/normalization/Reshape_1/ReadVariableOpг
#model/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#model/normalization/Reshape_1/shape▐
model/normalization/Reshape_1Reshape4model/normalization/Reshape_1/ReadVariableOp:value:0,model/normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
model/normalization/Reshape_1в
model/normalization/subSubinput_1$model/normalization/Reshape:output:0*
T0*/
_output_shapes
:         @@2
model/normalization/subХ
model/normalization/SqrtSqrt&model/normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
model/normalization/Sqrt║
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Sqrt:y:0*
T0*/
_output_shapes
:         @@2
model/normalization/truediv╝
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpу
model/conv2d/Conv2DConv2Dmodel/normalization/truediv:z:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
model/conv2d/Conv2D│
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp╝
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
model/conv2d/BiasAddЗ
model/conv2d/SeluSelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:           @2
model/conv2d/Selu°
6model/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp?model_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype028
6model/separable_conv2d/separable_conv2d/ReadVariableOp 
8model/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpAmodel_separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype02:
8model/separable_conv2d/separable_conv2d/ReadVariableOp_1╖
-model/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2/
-model/separable_conv2d/separable_conv2d/Shape┐
5model/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/separable_conv2d/separable_conv2d/dilation_rate┬
1model/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/conv2d/Selu:activations:0>model/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
23
1model/separable_conv2d/separable_conv2d/depthwise╛
'model/separable_conv2d/separable_conv2dConv2D:model/separable_conv2d/separable_conv2d/depthwise:output:0@model/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2)
'model/separable_conv2d/separable_conv2d╥
-model/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model/separable_conv2d/BiasAdd/ReadVariableOpя
model/separable_conv2d/BiasAddBiasAdd0model/separable_conv2d/separable_conv2d:output:05model/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d/BiasAddж
model/separable_conv2d/SeluSelu'model/separable_conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d/Selu 
8model/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_1/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_1/separable_conv2d/Shape├
7model/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_1/separable_conv2d/dilation_rate╙
3model/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative)model/separable_conv2d/Selu:activations:0@model/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_1/separable_conv2d/depthwise╞
)model/separable_conv2d_1/separable_conv2dConv2D<model/separable_conv2d_1/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_1/separable_conv2d╪
/model/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_1/BiasAdd/ReadVariableOpў
 model/separable_conv2d_1/BiasAddBiasAdd2model/separable_conv2d_1/separable_conv2d:output:07model/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_1/BiasAddм
model/separable_conv2d_1/SeluSelu)model/separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_1/Selu├
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOpъ
model/conv2d_1/Conv2DConv2Dmodel/conv2d/Selu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
model/conv2d_1/Conv2D║
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp┼
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
model/conv2d_1/BiasAdd░
model/add/addAddV2+model/separable_conv2d_1/Selu:activations:0model/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/add/add 
8model/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_2/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_2/separable_conv2d/Shape├
7model/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_2/separable_conv2d/dilation_rate╗
3model/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add/add:z:0@model/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_2/separable_conv2d/depthwise╞
)model/separable_conv2d_2/separable_conv2dConv2D<model/separable_conv2d_2/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_2/separable_conv2d╪
/model/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_2/BiasAdd/ReadVariableOpў
 model/separable_conv2d_2/BiasAddBiasAdd2model/separable_conv2d_2/separable_conv2d:output:07model/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_2/BiasAddм
model/separable_conv2d_2/SeluSelu)model/separable_conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_2/Selu 
8model/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_3_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_3/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_3/separable_conv2d/Shape├
7model/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_3/separable_conv2d/dilation_rate╒
3model/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative+model/separable_conv2d_2/Selu:activations:0@model/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_3/separable_conv2d/depthwise╞
)model/separable_conv2d_3/separable_conv2dConv2D<model/separable_conv2d_3/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_3/separable_conv2d╪
/model/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_3/BiasAdd/ReadVariableOpў
 model/separable_conv2d_3/BiasAddBiasAdd2model/separable_conv2d_3/separable_conv2d:output:07model/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_3/BiasAddм
model/separable_conv2d_3/SeluSelu)model/separable_conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_3/Seluж
model/add_1/addAddV2+model/separable_conv2d_3/Selu:activations:0model/add/add:z:0*
T0*0
_output_shapes
:           А2
model/add_1/add 
8model/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_4/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_4/separable_conv2d/Shape├
7model/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_4/separable_conv2d/dilation_rate╜
3model/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_1/add:z:0@model/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_4/separable_conv2d/depthwise╞
)model/separable_conv2d_4/separable_conv2dConv2D<model/separable_conv2d_4/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_4/separable_conv2d╪
/model/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_4/BiasAdd/ReadVariableOpў
 model/separable_conv2d_4/BiasAddBiasAdd2model/separable_conv2d_4/separable_conv2d:output:07model/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_4/BiasAddм
model/separable_conv2d_4/SeluSelu)model/separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_4/Selu 
8model/separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_5/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_5/separable_conv2d/Shape├
7model/separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_5/separable_conv2d/dilation_rate╒
3model/separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative+model/separable_conv2d_4/Selu:activations:0@model/separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_5/separable_conv2d/depthwise╞
)model/separable_conv2d_5/separable_conv2dConv2D<model/separable_conv2d_5/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_5/separable_conv2d╪
/model/separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_5/BiasAdd/ReadVariableOpў
 model/separable_conv2d_5/BiasAddBiasAdd2model/separable_conv2d_5/separable_conv2d:output:07model/separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_5/BiasAddм
model/separable_conv2d_5/SeluSelu)model/separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_5/Seluи
model/add_2/addAddV2+model/separable_conv2d_5/Selu:activations:0model/add_1/add:z:0*
T0*0
_output_shapes
:           А2
model/add_2/add 
8model/separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_6_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_6/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_6_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_6/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_6/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_6/separable_conv2d/Shape├
7model/separable_conv2d_6/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_6/separable_conv2d/dilation_rate╜
3model/separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_2/add:z:0@model/separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_6/separable_conv2d/depthwise╞
)model/separable_conv2d_6/separable_conv2dConv2D<model/separable_conv2d_6/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_6/separable_conv2d╪
/model/separable_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_6/BiasAdd/ReadVariableOpў
 model/separable_conv2d_6/BiasAddBiasAdd2model/separable_conv2d_6/separable_conv2d:output:07model/separable_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_6/BiasAddм
model/separable_conv2d_6/SeluSelu)model/separable_conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_6/Selu 
8model/separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_7_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_7/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_7_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_7/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_7/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_7/separable_conv2d/Shape├
7model/separable_conv2d_7/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_7/separable_conv2d/dilation_rate╒
3model/separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNative+model/separable_conv2d_6/Selu:activations:0@model/separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_7/separable_conv2d/depthwise╞
)model/separable_conv2d_7/separable_conv2dConv2D<model/separable_conv2d_7/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_7/separable_conv2d╪
/model/separable_conv2d_7/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_7/BiasAdd/ReadVariableOpў
 model/separable_conv2d_7/BiasAddBiasAdd2model/separable_conv2d_7/separable_conv2d:output:07model/separable_conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_7/BiasAddм
model/separable_conv2d_7/SeluSelu)model/separable_conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_7/Seluи
model/add_3/addAddV2+model/separable_conv2d_7/Selu:activations:0model/add_2/add:z:0*
T0*0
_output_shapes
:           А2
model/add_3/add 
8model/separable_conv2d_8/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_8_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_8/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_8/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_8_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_8/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_8/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_8/separable_conv2d/Shape├
7model/separable_conv2d_8/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_8/separable_conv2d/dilation_rate╜
3model/separable_conv2d_8/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_3/add:z:0@model/separable_conv2d_8/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_8/separable_conv2d/depthwise╞
)model/separable_conv2d_8/separable_conv2dConv2D<model/separable_conv2d_8/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_8/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_8/separable_conv2d╪
/model/separable_conv2d_8/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_8/BiasAdd/ReadVariableOpў
 model/separable_conv2d_8/BiasAddBiasAdd2model/separable_conv2d_8/separable_conv2d:output:07model/separable_conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_8/BiasAddм
model/separable_conv2d_8/SeluSelu)model/separable_conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_8/Selu 
8model/separable_conv2d_9/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_9_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_9/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_9/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_9_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_9/separable_conv2d/ReadVariableOp_1╗
/model/separable_conv2d_9/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_9/separable_conv2d/Shape├
7model/separable_conv2d_9/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_9/separable_conv2d/dilation_rate╒
3model/separable_conv2d_9/separable_conv2d/depthwiseDepthwiseConv2dNative+model/separable_conv2d_8/Selu:activations:0@model/separable_conv2d_9/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
25
3model/separable_conv2d_9/separable_conv2d/depthwise╞
)model/separable_conv2d_9/separable_conv2dConv2D<model/separable_conv2d_9/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_9/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2+
)model/separable_conv2d_9/separable_conv2d╪
/model/separable_conv2d_9/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_9/BiasAdd/ReadVariableOpў
 model/separable_conv2d_9/BiasAddBiasAdd2model/separable_conv2d_9/separable_conv2d:output:07model/separable_conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2"
 model/separable_conv2d_9/BiasAddм
model/separable_conv2d_9/SeluSelu)model/separable_conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:           А2
model/separable_conv2d_9/Seluи
model/add_4/addAddV2+model/separable_conv2d_9/Selu:activations:0model/add_3/add:z:0*
T0*0
_output_shapes
:           А2
model/add_4/addВ
9model/separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOpBmodel_separable_conv2d_10_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02;
9model/separable_conv2d_10/separable_conv2d/ReadVariableOpЙ
;model/separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOpDmodel_separable_conv2d_10_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02=
;model/separable_conv2d_10/separable_conv2d/ReadVariableOp_1╜
0model/separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model/separable_conv2d_10/separable_conv2d/Shape┼
8model/separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model/separable_conv2d_10/separable_conv2d/dilation_rate└
4model/separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_4/add:z:0Amodel/separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
26
4model/separable_conv2d_10/separable_conv2d/depthwise╩
*model/separable_conv2d_10/separable_conv2dConv2D=model/separable_conv2d_10/separable_conv2d/depthwise:output:0Cmodel/separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2,
*model/separable_conv2d_10/separable_conv2d█
0model/separable_conv2d_10/BiasAdd/ReadVariableOpReadVariableOp9model_separable_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/separable_conv2d_10/BiasAdd/ReadVariableOp√
!model/separable_conv2d_10/BiasAddBiasAdd3model/separable_conv2d_10/separable_conv2d:output:08model/separable_conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2#
!model/separable_conv2d_10/BiasAddп
model/separable_conv2d_10/SeluSelu*model/separable_conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d_10/SeluВ
9model/separable_conv2d_11/separable_conv2d/ReadVariableOpReadVariableOpBmodel_separable_conv2d_11_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02;
9model/separable_conv2d_11/separable_conv2d/ReadVariableOpЙ
;model/separable_conv2d_11/separable_conv2d/ReadVariableOp_1ReadVariableOpDmodel_separable_conv2d_11_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02=
;model/separable_conv2d_11/separable_conv2d/ReadVariableOp_1╜
0model/separable_conv2d_11/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model/separable_conv2d_11/separable_conv2d/Shape┼
8model/separable_conv2d_11/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model/separable_conv2d_11/separable_conv2d/dilation_rate┘
4model/separable_conv2d_11/separable_conv2d/depthwiseDepthwiseConv2dNative,model/separable_conv2d_10/Selu:activations:0Amodel/separable_conv2d_11/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
26
4model/separable_conv2d_11/separable_conv2d/depthwise╩
*model/separable_conv2d_11/separable_conv2dConv2D=model/separable_conv2d_11/separable_conv2d/depthwise:output:0Cmodel/separable_conv2d_11/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2,
*model/separable_conv2d_11/separable_conv2d█
0model/separable_conv2d_11/BiasAdd/ReadVariableOpReadVariableOp9model_separable_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/separable_conv2d_11/BiasAdd/ReadVariableOp√
!model/separable_conv2d_11/BiasAddBiasAdd3model/separable_conv2d_11/separable_conv2d:output:08model/separable_conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2#
!model/separable_conv2d_11/BiasAddп
model/separable_conv2d_11/SeluSelu*model/separable_conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d_11/Seluй
model/add_5/addAddV2,model/separable_conv2d_11/Selu:activations:0model/add_4/add:z:0*
T0*0
_output_shapes
:           А2
model/add_5/addВ
9model/separable_conv2d_12/separable_conv2d/ReadVariableOpReadVariableOpBmodel_separable_conv2d_12_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02;
9model/separable_conv2d_12/separable_conv2d/ReadVariableOpЙ
;model/separable_conv2d_12/separable_conv2d/ReadVariableOp_1ReadVariableOpDmodel_separable_conv2d_12_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02=
;model/separable_conv2d_12/separable_conv2d/ReadVariableOp_1╜
0model/separable_conv2d_12/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model/separable_conv2d_12/separable_conv2d/Shape┼
8model/separable_conv2d_12/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model/separable_conv2d_12/separable_conv2d/dilation_rate└
4model/separable_conv2d_12/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_5/add:z:0Amodel/separable_conv2d_12/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
26
4model/separable_conv2d_12/separable_conv2d/depthwise╩
*model/separable_conv2d_12/separable_conv2dConv2D=model/separable_conv2d_12/separable_conv2d/depthwise:output:0Cmodel/separable_conv2d_12/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2,
*model/separable_conv2d_12/separable_conv2d█
0model/separable_conv2d_12/BiasAdd/ReadVariableOpReadVariableOp9model_separable_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/separable_conv2d_12/BiasAdd/ReadVariableOp√
!model/separable_conv2d_12/BiasAddBiasAdd3model/separable_conv2d_12/separable_conv2d:output:08model/separable_conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2#
!model/separable_conv2d_12/BiasAddп
model/separable_conv2d_12/SeluSelu*model/separable_conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d_12/SeluВ
9model/separable_conv2d_13/separable_conv2d/ReadVariableOpReadVariableOpBmodel_separable_conv2d_13_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02;
9model/separable_conv2d_13/separable_conv2d/ReadVariableOpЙ
;model/separable_conv2d_13/separable_conv2d/ReadVariableOp_1ReadVariableOpDmodel_separable_conv2d_13_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02=
;model/separable_conv2d_13/separable_conv2d/ReadVariableOp_1╜
0model/separable_conv2d_13/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model/separable_conv2d_13/separable_conv2d/Shape┼
8model/separable_conv2d_13/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model/separable_conv2d_13/separable_conv2d/dilation_rate┘
4model/separable_conv2d_13/separable_conv2d/depthwiseDepthwiseConv2dNative,model/separable_conv2d_12/Selu:activations:0Amodel/separable_conv2d_13/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
26
4model/separable_conv2d_13/separable_conv2d/depthwise╩
*model/separable_conv2d_13/separable_conv2dConv2D=model/separable_conv2d_13/separable_conv2d/depthwise:output:0Cmodel/separable_conv2d_13/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2,
*model/separable_conv2d_13/separable_conv2d█
0model/separable_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp9model_separable_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/separable_conv2d_13/BiasAdd/ReadVariableOp√
!model/separable_conv2d_13/BiasAddBiasAdd3model/separable_conv2d_13/separable_conv2d:output:08model/separable_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2#
!model/separable_conv2d_13/BiasAddп
model/separable_conv2d_13/SeluSelu*model/separable_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d_13/Seluй
model/add_6/addAddV2,model/separable_conv2d_13/Selu:activations:0model/add_5/add:z:0*
T0*0
_output_shapes
:           А2
model/add_6/addВ
9model/separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOpBmodel_separable_conv2d_14_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02;
9model/separable_conv2d_14/separable_conv2d/ReadVariableOpЙ
;model/separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOpDmodel_separable_conv2d_14_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02=
;model/separable_conv2d_14/separable_conv2d/ReadVariableOp_1╜
0model/separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model/separable_conv2d_14/separable_conv2d/Shape┼
8model/separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model/separable_conv2d_14/separable_conv2d/dilation_rate└
4model/separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_6/add:z:0Amodel/separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
26
4model/separable_conv2d_14/separable_conv2d/depthwise╩
*model/separable_conv2d_14/separable_conv2dConv2D=model/separable_conv2d_14/separable_conv2d/depthwise:output:0Cmodel/separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2,
*model/separable_conv2d_14/separable_conv2d█
0model/separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp9model_separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/separable_conv2d_14/BiasAdd/ReadVariableOp√
!model/separable_conv2d_14/BiasAddBiasAdd3model/separable_conv2d_14/separable_conv2d:output:08model/separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2#
!model/separable_conv2d_14/BiasAddп
model/separable_conv2d_14/SeluSelu*model/separable_conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d_14/SeluВ
9model/separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOpBmodel_separable_conv2d_15_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02;
9model/separable_conv2d_15/separable_conv2d/ReadVariableOpЙ
;model/separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOpDmodel_separable_conv2d_15_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02=
;model/separable_conv2d_15/separable_conv2d/ReadVariableOp_1╜
0model/separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model/separable_conv2d_15/separable_conv2d/Shape┼
8model/separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2:
8model/separable_conv2d_15/separable_conv2d/dilation_rate┘
4model/separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative,model/separable_conv2d_14/Selu:activations:0Amodel/separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
26
4model/separable_conv2d_15/separable_conv2d/depthwise╩
*model/separable_conv2d_15/separable_conv2dConv2D=model/separable_conv2d_15/separable_conv2d/depthwise:output:0Cmodel/separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2,
*model/separable_conv2d_15/separable_conv2d█
0model/separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp9model_separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model/separable_conv2d_15/BiasAdd/ReadVariableOp√
!model/separable_conv2d_15/BiasAddBiasAdd3model/separable_conv2d_15/separable_conv2d:output:08model/separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2#
!model/separable_conv2d_15/BiasAddп
model/separable_conv2d_15/SeluSelu*model/separable_conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:           А2 
model/separable_conv2d_15/Seluр
model/max_pooling2d/MaxPoolMaxPool,model/separable_conv2d_15/Selu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
model/max_pooling2d/MaxPool─
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp▐
model/conv2d_2/Conv2DConv2Dmodel/add_6/add:z:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
model/conv2d_2/Conv2D║
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp┼
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
model/conv2d_2/BiasAddн
model/add_7/addAddV2$model/max_pooling2d/MaxPool:output:0model/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
model/add_7/add┐
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices┌
#model/global_average_pooling2d/MeanMeanmodel/add_7/add:z:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2%
#model/global_average_pooling2d/Mean╓
-model/regression_head_1/MatMul/ReadVariableOpReadVariableOp6model_regression_head_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-model/regression_head_1/MatMul/ReadVariableOpс
model/regression_head_1/MatMulMatMul,model/global_average_pooling2d/Mean:output:05model/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
model/regression_head_1/MatMul╘
.model/regression_head_1/BiasAdd/ReadVariableOpReadVariableOp7model_regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model/regression_head_1/BiasAdd/ReadVariableOpс
model/regression_head_1/BiasAddBiasAdd(model/regression_head_1/MatMul:product:06model/regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2!
model/regression_head_1/BiasAdd╬
IdentityIdentity(model/regression_head_1/BiasAdd:output:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp/^model/regression_head_1/BiasAdd/ReadVariableOp.^model/regression_head_1/MatMul/ReadVariableOp.^model/separable_conv2d/BiasAdd/ReadVariableOp7^model/separable_conv2d/separable_conv2d/ReadVariableOp9^model/separable_conv2d/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_1/BiasAdd/ReadVariableOp9^model/separable_conv2d_1/separable_conv2d/ReadVariableOp;^model/separable_conv2d_1/separable_conv2d/ReadVariableOp_11^model/separable_conv2d_10/BiasAdd/ReadVariableOp:^model/separable_conv2d_10/separable_conv2d/ReadVariableOp<^model/separable_conv2d_10/separable_conv2d/ReadVariableOp_11^model/separable_conv2d_11/BiasAdd/ReadVariableOp:^model/separable_conv2d_11/separable_conv2d/ReadVariableOp<^model/separable_conv2d_11/separable_conv2d/ReadVariableOp_11^model/separable_conv2d_12/BiasAdd/ReadVariableOp:^model/separable_conv2d_12/separable_conv2d/ReadVariableOp<^model/separable_conv2d_12/separable_conv2d/ReadVariableOp_11^model/separable_conv2d_13/BiasAdd/ReadVariableOp:^model/separable_conv2d_13/separable_conv2d/ReadVariableOp<^model/separable_conv2d_13/separable_conv2d/ReadVariableOp_11^model/separable_conv2d_14/BiasAdd/ReadVariableOp:^model/separable_conv2d_14/separable_conv2d/ReadVariableOp<^model/separable_conv2d_14/separable_conv2d/ReadVariableOp_11^model/separable_conv2d_15/BiasAdd/ReadVariableOp:^model/separable_conv2d_15/separable_conv2d/ReadVariableOp<^model/separable_conv2d_15/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_2/BiasAdd/ReadVariableOp9^model/separable_conv2d_2/separable_conv2d/ReadVariableOp;^model/separable_conv2d_2/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_3/BiasAdd/ReadVariableOp9^model/separable_conv2d_3/separable_conv2d/ReadVariableOp;^model/separable_conv2d_3/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_4/BiasAdd/ReadVariableOp9^model/separable_conv2d_4/separable_conv2d/ReadVariableOp;^model/separable_conv2d_4/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_5/BiasAdd/ReadVariableOp9^model/separable_conv2d_5/separable_conv2d/ReadVariableOp;^model/separable_conv2d_5/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_6/BiasAdd/ReadVariableOp9^model/separable_conv2d_6/separable_conv2d/ReadVariableOp;^model/separable_conv2d_6/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_7/BiasAdd/ReadVariableOp9^model/separable_conv2d_7/separable_conv2d/ReadVariableOp;^model/separable_conv2d_7/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_8/BiasAdd/ReadVariableOp9^model/separable_conv2d_8/separable_conv2d/ReadVariableOp;^model/separable_conv2d_8/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_9/BiasAdd/ReadVariableOp9^model/separable_conv2d_9/separable_conv2d/ReadVariableOp;^model/separable_conv2d_9/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2X
*model/normalization/Reshape/ReadVariableOp*model/normalization/Reshape/ReadVariableOp2\
,model/normalization/Reshape_1/ReadVariableOp,model/normalization/Reshape_1/ReadVariableOp2`
.model/regression_head_1/BiasAdd/ReadVariableOp.model/regression_head_1/BiasAdd/ReadVariableOp2^
-model/regression_head_1/MatMul/ReadVariableOp-model/regression_head_1/MatMul/ReadVariableOp2^
-model/separable_conv2d/BiasAdd/ReadVariableOp-model/separable_conv2d/BiasAdd/ReadVariableOp2p
6model/separable_conv2d/separable_conv2d/ReadVariableOp6model/separable_conv2d/separable_conv2d/ReadVariableOp2t
8model/separable_conv2d/separable_conv2d/ReadVariableOp_18model/separable_conv2d/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_1/BiasAdd/ReadVariableOp/model/separable_conv2d_1/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_1/separable_conv2d/ReadVariableOp8model/separable_conv2d_1/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_12d
0model/separable_conv2d_10/BiasAdd/ReadVariableOp0model/separable_conv2d_10/BiasAdd/ReadVariableOp2v
9model/separable_conv2d_10/separable_conv2d/ReadVariableOp9model/separable_conv2d_10/separable_conv2d/ReadVariableOp2z
;model/separable_conv2d_10/separable_conv2d/ReadVariableOp_1;model/separable_conv2d_10/separable_conv2d/ReadVariableOp_12d
0model/separable_conv2d_11/BiasAdd/ReadVariableOp0model/separable_conv2d_11/BiasAdd/ReadVariableOp2v
9model/separable_conv2d_11/separable_conv2d/ReadVariableOp9model/separable_conv2d_11/separable_conv2d/ReadVariableOp2z
;model/separable_conv2d_11/separable_conv2d/ReadVariableOp_1;model/separable_conv2d_11/separable_conv2d/ReadVariableOp_12d
0model/separable_conv2d_12/BiasAdd/ReadVariableOp0model/separable_conv2d_12/BiasAdd/ReadVariableOp2v
9model/separable_conv2d_12/separable_conv2d/ReadVariableOp9model/separable_conv2d_12/separable_conv2d/ReadVariableOp2z
;model/separable_conv2d_12/separable_conv2d/ReadVariableOp_1;model/separable_conv2d_12/separable_conv2d/ReadVariableOp_12d
0model/separable_conv2d_13/BiasAdd/ReadVariableOp0model/separable_conv2d_13/BiasAdd/ReadVariableOp2v
9model/separable_conv2d_13/separable_conv2d/ReadVariableOp9model/separable_conv2d_13/separable_conv2d/ReadVariableOp2z
;model/separable_conv2d_13/separable_conv2d/ReadVariableOp_1;model/separable_conv2d_13/separable_conv2d/ReadVariableOp_12d
0model/separable_conv2d_14/BiasAdd/ReadVariableOp0model/separable_conv2d_14/BiasAdd/ReadVariableOp2v
9model/separable_conv2d_14/separable_conv2d/ReadVariableOp9model/separable_conv2d_14/separable_conv2d/ReadVariableOp2z
;model/separable_conv2d_14/separable_conv2d/ReadVariableOp_1;model/separable_conv2d_14/separable_conv2d/ReadVariableOp_12d
0model/separable_conv2d_15/BiasAdd/ReadVariableOp0model/separable_conv2d_15/BiasAdd/ReadVariableOp2v
9model/separable_conv2d_15/separable_conv2d/ReadVariableOp9model/separable_conv2d_15/separable_conv2d/ReadVariableOp2z
;model/separable_conv2d_15/separable_conv2d/ReadVariableOp_1;model/separable_conv2d_15/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_2/BiasAdd/ReadVariableOp/model/separable_conv2d_2/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_2/separable_conv2d/ReadVariableOp8model/separable_conv2d_2/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_3/BiasAdd/ReadVariableOp/model/separable_conv2d_3/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_3/separable_conv2d/ReadVariableOp8model/separable_conv2d_3/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_4/BiasAdd/ReadVariableOp/model/separable_conv2d_4/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_4/separable_conv2d/ReadVariableOp8model/separable_conv2d_4/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_5/BiasAdd/ReadVariableOp/model/separable_conv2d_5/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_5/separable_conv2d/ReadVariableOp8model/separable_conv2d_5/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_6/BiasAdd/ReadVariableOp/model/separable_conv2d_6/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_6/separable_conv2d/ReadVariableOp8model/separable_conv2d_6/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_6/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_6/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_7/BiasAdd/ReadVariableOp/model/separable_conv2d_7/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_7/separable_conv2d/ReadVariableOp8model/separable_conv2d_7/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_7/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_7/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_8/BiasAdd/ReadVariableOp/model/separable_conv2d_8/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_8/separable_conv2d/ReadVariableOp8model/separable_conv2d_8/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_8/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_8/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_9/BiasAdd/ReadVariableOp/model/separable_conv2d_9/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_9/separable_conv2d/ReadVariableOp8model/separable_conv2d_9/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_9/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_9/separable_conv2d/ReadVariableOp_1:' #
!
_user_specified_name	input_1
°

▄
C__inference_conv2d_2_layer_call_and_return_conditional_losses_80013

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ё
j
@__inference_add_3_layer_call_and_return_conditional_losses_80154

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
г
╪
3__inference_separable_conv2d_12_layer_call_fn_79911

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_799022
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ё
j
@__inference_add_5_layer_call_and_return_conditional_losses_80200

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
╬
Q
%__inference_add_3_layer_call_fn_81435
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:           А*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_801542
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
╗
╬
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_79596

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ё
j
@__inference_add_6_layer_call_and_return_conditional_losses_80223

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:           А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:           А:           А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
э
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_80028

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
■┌
Аk
!__inference__traced_restore_82574
file_prefix'
#assignvariableop_normalization_mean-
)assignvariableop_1_normalization_variance*
&assignvariableop_2_normalization_count$
 assignvariableop_3_conv2d_kernel"
assignvariableop_4_conv2d_bias8
4assignvariableop_5_separable_conv2d_depthwise_kernel8
4assignvariableop_6_separable_conv2d_pointwise_kernel,
(assignvariableop_7_separable_conv2d_bias:
6assignvariableop_8_separable_conv2d_1_depthwise_kernel:
6assignvariableop_9_separable_conv2d_1_pointwise_kernel/
+assignvariableop_10_separable_conv2d_1_bias'
#assignvariableop_11_conv2d_1_kernel%
!assignvariableop_12_conv2d_1_bias;
7assignvariableop_13_separable_conv2d_2_depthwise_kernel;
7assignvariableop_14_separable_conv2d_2_pointwise_kernel/
+assignvariableop_15_separable_conv2d_2_bias;
7assignvariableop_16_separable_conv2d_3_depthwise_kernel;
7assignvariableop_17_separable_conv2d_3_pointwise_kernel/
+assignvariableop_18_separable_conv2d_3_bias;
7assignvariableop_19_separable_conv2d_4_depthwise_kernel;
7assignvariableop_20_separable_conv2d_4_pointwise_kernel/
+assignvariableop_21_separable_conv2d_4_bias;
7assignvariableop_22_separable_conv2d_5_depthwise_kernel;
7assignvariableop_23_separable_conv2d_5_pointwise_kernel/
+assignvariableop_24_separable_conv2d_5_bias;
7assignvariableop_25_separable_conv2d_6_depthwise_kernel;
7assignvariableop_26_separable_conv2d_6_pointwise_kernel/
+assignvariableop_27_separable_conv2d_6_bias;
7assignvariableop_28_separable_conv2d_7_depthwise_kernel;
7assignvariableop_29_separable_conv2d_7_pointwise_kernel/
+assignvariableop_30_separable_conv2d_7_bias;
7assignvariableop_31_separable_conv2d_8_depthwise_kernel;
7assignvariableop_32_separable_conv2d_8_pointwise_kernel/
+assignvariableop_33_separable_conv2d_8_bias;
7assignvariableop_34_separable_conv2d_9_depthwise_kernel;
7assignvariableop_35_separable_conv2d_9_pointwise_kernel/
+assignvariableop_36_separable_conv2d_9_bias<
8assignvariableop_37_separable_conv2d_10_depthwise_kernel<
8assignvariableop_38_separable_conv2d_10_pointwise_kernel0
,assignvariableop_39_separable_conv2d_10_bias<
8assignvariableop_40_separable_conv2d_11_depthwise_kernel<
8assignvariableop_41_separable_conv2d_11_pointwise_kernel0
,assignvariableop_42_separable_conv2d_11_bias<
8assignvariableop_43_separable_conv2d_12_depthwise_kernel<
8assignvariableop_44_separable_conv2d_12_pointwise_kernel0
,assignvariableop_45_separable_conv2d_12_bias<
8assignvariableop_46_separable_conv2d_13_depthwise_kernel<
8assignvariableop_47_separable_conv2d_13_pointwise_kernel0
,assignvariableop_48_separable_conv2d_13_bias<
8assignvariableop_49_separable_conv2d_14_depthwise_kernel<
8assignvariableop_50_separable_conv2d_14_pointwise_kernel0
,assignvariableop_51_separable_conv2d_14_bias<
8assignvariableop_52_separable_conv2d_15_depthwise_kernel<
8assignvariableop_53_separable_conv2d_15_pointwise_kernel0
,assignvariableop_54_separable_conv2d_15_bias'
#assignvariableop_55_conv2d_2_kernel%
!assignvariableop_56_conv2d_2_bias0
,assignvariableop_57_regression_head_1_kernel.
*assignvariableop_58_regression_head_1_bias
assignvariableop_59_total
assignvariableop_60_count'
#assignvariableop_61_conv2d_kernel_m%
!assignvariableop_62_conv2d_bias_m;
7assignvariableop_63_separable_conv2d_depthwise_kernel_m;
7assignvariableop_64_separable_conv2d_pointwise_kernel_m/
+assignvariableop_65_separable_conv2d_bias_m=
9assignvariableop_66_separable_conv2d_1_depthwise_kernel_m=
9assignvariableop_67_separable_conv2d_1_pointwise_kernel_m1
-assignvariableop_68_separable_conv2d_1_bias_m)
%assignvariableop_69_conv2d_1_kernel_m'
#assignvariableop_70_conv2d_1_bias_m=
9assignvariableop_71_separable_conv2d_2_depthwise_kernel_m=
9assignvariableop_72_separable_conv2d_2_pointwise_kernel_m1
-assignvariableop_73_separable_conv2d_2_bias_m=
9assignvariableop_74_separable_conv2d_3_depthwise_kernel_m=
9assignvariableop_75_separable_conv2d_3_pointwise_kernel_m1
-assignvariableop_76_separable_conv2d_3_bias_m=
9assignvariableop_77_separable_conv2d_4_depthwise_kernel_m=
9assignvariableop_78_separable_conv2d_4_pointwise_kernel_m1
-assignvariableop_79_separable_conv2d_4_bias_m=
9assignvariableop_80_separable_conv2d_5_depthwise_kernel_m=
9assignvariableop_81_separable_conv2d_5_pointwise_kernel_m1
-assignvariableop_82_separable_conv2d_5_bias_m=
9assignvariableop_83_separable_conv2d_6_depthwise_kernel_m=
9assignvariableop_84_separable_conv2d_6_pointwise_kernel_m1
-assignvariableop_85_separable_conv2d_6_bias_m=
9assignvariableop_86_separable_conv2d_7_depthwise_kernel_m=
9assignvariableop_87_separable_conv2d_7_pointwise_kernel_m1
-assignvariableop_88_separable_conv2d_7_bias_m=
9assignvariableop_89_separable_conv2d_8_depthwise_kernel_m=
9assignvariableop_90_separable_conv2d_8_pointwise_kernel_m1
-assignvariableop_91_separable_conv2d_8_bias_m=
9assignvariableop_92_separable_conv2d_9_depthwise_kernel_m=
9assignvariableop_93_separable_conv2d_9_pointwise_kernel_m1
-assignvariableop_94_separable_conv2d_9_bias_m>
:assignvariableop_95_separable_conv2d_10_depthwise_kernel_m>
:assignvariableop_96_separable_conv2d_10_pointwise_kernel_m2
.assignvariableop_97_separable_conv2d_10_bias_m>
:assignvariableop_98_separable_conv2d_11_depthwise_kernel_m>
:assignvariableop_99_separable_conv2d_11_pointwise_kernel_m3
/assignvariableop_100_separable_conv2d_11_bias_m?
;assignvariableop_101_separable_conv2d_12_depthwise_kernel_m?
;assignvariableop_102_separable_conv2d_12_pointwise_kernel_m3
/assignvariableop_103_separable_conv2d_12_bias_m?
;assignvariableop_104_separable_conv2d_13_depthwise_kernel_m?
;assignvariableop_105_separable_conv2d_13_pointwise_kernel_m3
/assignvariableop_106_separable_conv2d_13_bias_m?
;assignvariableop_107_separable_conv2d_14_depthwise_kernel_m?
;assignvariableop_108_separable_conv2d_14_pointwise_kernel_m3
/assignvariableop_109_separable_conv2d_14_bias_m?
;assignvariableop_110_separable_conv2d_15_depthwise_kernel_m?
;assignvariableop_111_separable_conv2d_15_pointwise_kernel_m3
/assignvariableop_112_separable_conv2d_15_bias_m*
&assignvariableop_113_conv2d_2_kernel_m(
$assignvariableop_114_conv2d_2_bias_m3
/assignvariableop_115_regression_head_1_kernel_m1
-assignvariableop_116_regression_head_1_bias_m(
$assignvariableop_117_conv2d_kernel_v&
"assignvariableop_118_conv2d_bias_v<
8assignvariableop_119_separable_conv2d_depthwise_kernel_v<
8assignvariableop_120_separable_conv2d_pointwise_kernel_v0
,assignvariableop_121_separable_conv2d_bias_v>
:assignvariableop_122_separable_conv2d_1_depthwise_kernel_v>
:assignvariableop_123_separable_conv2d_1_pointwise_kernel_v2
.assignvariableop_124_separable_conv2d_1_bias_v*
&assignvariableop_125_conv2d_1_kernel_v(
$assignvariableop_126_conv2d_1_bias_v>
:assignvariableop_127_separable_conv2d_2_depthwise_kernel_v>
:assignvariableop_128_separable_conv2d_2_pointwise_kernel_v2
.assignvariableop_129_separable_conv2d_2_bias_v>
:assignvariableop_130_separable_conv2d_3_depthwise_kernel_v>
:assignvariableop_131_separable_conv2d_3_pointwise_kernel_v2
.assignvariableop_132_separable_conv2d_3_bias_v>
:assignvariableop_133_separable_conv2d_4_depthwise_kernel_v>
:assignvariableop_134_separable_conv2d_4_pointwise_kernel_v2
.assignvariableop_135_separable_conv2d_4_bias_v>
:assignvariableop_136_separable_conv2d_5_depthwise_kernel_v>
:assignvariableop_137_separable_conv2d_5_pointwise_kernel_v2
.assignvariableop_138_separable_conv2d_5_bias_v>
:assignvariableop_139_separable_conv2d_6_depthwise_kernel_v>
:assignvariableop_140_separable_conv2d_6_pointwise_kernel_v2
.assignvariableop_141_separable_conv2d_6_bias_v>
:assignvariableop_142_separable_conv2d_7_depthwise_kernel_v>
:assignvariableop_143_separable_conv2d_7_pointwise_kernel_v2
.assignvariableop_144_separable_conv2d_7_bias_v>
:assignvariableop_145_separable_conv2d_8_depthwise_kernel_v>
:assignvariableop_146_separable_conv2d_8_pointwise_kernel_v2
.assignvariableop_147_separable_conv2d_8_bias_v>
:assignvariableop_148_separable_conv2d_9_depthwise_kernel_v>
:assignvariableop_149_separable_conv2d_9_pointwise_kernel_v2
.assignvariableop_150_separable_conv2d_9_bias_v?
;assignvariableop_151_separable_conv2d_10_depthwise_kernel_v?
;assignvariableop_152_separable_conv2d_10_pointwise_kernel_v3
/assignvariableop_153_separable_conv2d_10_bias_v?
;assignvariableop_154_separable_conv2d_11_depthwise_kernel_v?
;assignvariableop_155_separable_conv2d_11_pointwise_kernel_v3
/assignvariableop_156_separable_conv2d_11_bias_v?
;assignvariableop_157_separable_conv2d_12_depthwise_kernel_v?
;assignvariableop_158_separable_conv2d_12_pointwise_kernel_v3
/assignvariableop_159_separable_conv2d_12_bias_v?
;assignvariableop_160_separable_conv2d_13_depthwise_kernel_v?
;assignvariableop_161_separable_conv2d_13_pointwise_kernel_v3
/assignvariableop_162_separable_conv2d_13_bias_v?
;assignvariableop_163_separable_conv2d_14_depthwise_kernel_v?
;assignvariableop_164_separable_conv2d_14_pointwise_kernel_v3
/assignvariableop_165_separable_conv2d_14_bias_v?
;assignvariableop_166_separable_conv2d_15_depthwise_kernel_v?
;assignvariableop_167_separable_conv2d_15_pointwise_kernel_v3
/assignvariableop_168_separable_conv2d_15_bias_v*
&assignvariableop_169_conv2d_2_kernel_v(
$assignvariableop_170_conv2d_2_bias_v3
/assignvariableop_171_regression_head_1_kernel_v1
-assignvariableop_172_regression_head_1_bias_v
identity_174ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_110вAssignVariableOp_111вAssignVariableOp_112вAssignVariableOp_113вAssignVariableOp_114вAssignVariableOp_115вAssignVariableOp_116вAssignVariableOp_117вAssignVariableOp_118вAssignVariableOp_119вAssignVariableOp_12вAssignVariableOp_120вAssignVariableOp_121вAssignVariableOp_122вAssignVariableOp_123вAssignVariableOp_124вAssignVariableOp_125вAssignVariableOp_126вAssignVariableOp_127вAssignVariableOp_128вAssignVariableOp_129вAssignVariableOp_13вAssignVariableOp_130вAssignVariableOp_131вAssignVariableOp_132вAssignVariableOp_133вAssignVariableOp_134вAssignVariableOp_135вAssignVariableOp_136вAssignVariableOp_137вAssignVariableOp_138вAssignVariableOp_139вAssignVariableOp_14вAssignVariableOp_140вAssignVariableOp_141вAssignVariableOp_142вAssignVariableOp_143вAssignVariableOp_144вAssignVariableOp_145вAssignVariableOp_146вAssignVariableOp_147вAssignVariableOp_148вAssignVariableOp_149вAssignVariableOp_15вAssignVariableOp_150вAssignVariableOp_151вAssignVariableOp_152вAssignVariableOp_153вAssignVariableOp_154вAssignVariableOp_155вAssignVariableOp_156вAssignVariableOp_157вAssignVariableOp_158вAssignVariableOp_159вAssignVariableOp_16вAssignVariableOp_160вAssignVariableOp_161вAssignVariableOp_162вAssignVariableOp_163вAssignVariableOp_164вAssignVariableOp_165вAssignVariableOp_166вAssignVariableOp_167вAssignVariableOp_168вAssignVariableOp_169вAssignVariableOp_17вAssignVariableOp_170вAssignVariableOp_171вAssignVariableOp_172вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99в	RestoreV2вRestoreV2_1├l
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:н*
dtype0*╬k
value─kB┴kнB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-11/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-13/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-15/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-18/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-11/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-13/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-15/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-16/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-18/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesэ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:н*
dtype0*Ё
valueцBунB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesУ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*╛
dtypes│
░2н2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityУ
AssignVariableOpAssignVariableOp#assignvariableop_normalization_meanIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Я
AssignVariableOp_1AssignVariableOp)assignvariableop_1_normalization_varianceIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ь
AssignVariableOp_2AssignVariableOp&assignvariableop_2_normalization_countIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ф
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv2d_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5к
AssignVariableOp_5AssignVariableOp4assignvariableop_5_separable_conv2d_depthwise_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp4assignvariableop_6_separable_conv2d_pointwise_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ю
AssignVariableOp_7AssignVariableOp(assignvariableop_7_separable_conv2d_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8м
AssignVariableOp_8AssignVariableOp6assignvariableop_8_separable_conv2d_1_depthwise_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9м
AssignVariableOp_9AssignVariableOp6assignvariableop_9_separable_conv2d_1_pointwise_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10д
AssignVariableOp_10AssignVariableOp+assignvariableop_10_separable_conv2d_1_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ь
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_1_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ъ
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_1_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13░
AssignVariableOp_13AssignVariableOp7assignvariableop_13_separable_conv2d_2_depthwise_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14░
AssignVariableOp_14AssignVariableOp7assignvariableop_14_separable_conv2d_2_pointwise_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15д
AssignVariableOp_15AssignVariableOp+assignvariableop_15_separable_conv2d_2_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16░
AssignVariableOp_16AssignVariableOp7assignvariableop_16_separable_conv2d_3_depthwise_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17░
AssignVariableOp_17AssignVariableOp7assignvariableop_17_separable_conv2d_3_pointwise_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18д
AssignVariableOp_18AssignVariableOp+assignvariableop_18_separable_conv2d_3_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19░
AssignVariableOp_19AssignVariableOp7assignvariableop_19_separable_conv2d_4_depthwise_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20░
AssignVariableOp_20AssignVariableOp7assignvariableop_20_separable_conv2d_4_pointwise_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21д
AssignVariableOp_21AssignVariableOp+assignvariableop_21_separable_conv2d_4_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22░
AssignVariableOp_22AssignVariableOp7assignvariableop_22_separable_conv2d_5_depthwise_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23░
AssignVariableOp_23AssignVariableOp7assignvariableop_23_separable_conv2d_5_pointwise_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24д
AssignVariableOp_24AssignVariableOp+assignvariableop_24_separable_conv2d_5_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25░
AssignVariableOp_25AssignVariableOp7assignvariableop_25_separable_conv2d_6_depthwise_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp7assignvariableop_26_separable_conv2d_6_pointwise_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27д
AssignVariableOp_27AssignVariableOp+assignvariableop_27_separable_conv2d_6_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28░
AssignVariableOp_28AssignVariableOp7assignvariableop_28_separable_conv2d_7_depthwise_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29░
AssignVariableOp_29AssignVariableOp7assignvariableop_29_separable_conv2d_7_pointwise_kernelIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30д
AssignVariableOp_30AssignVariableOp+assignvariableop_30_separable_conv2d_7_biasIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31░
AssignVariableOp_31AssignVariableOp7assignvariableop_31_separable_conv2d_8_depthwise_kernelIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32░
AssignVariableOp_32AssignVariableOp7assignvariableop_32_separable_conv2d_8_pointwise_kernelIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33д
AssignVariableOp_33AssignVariableOp+assignvariableop_33_separable_conv2d_8_biasIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34░
AssignVariableOp_34AssignVariableOp7assignvariableop_34_separable_conv2d_9_depthwise_kernelIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35░
AssignVariableOp_35AssignVariableOp7assignvariableop_35_separable_conv2d_9_pointwise_kernelIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36д
AssignVariableOp_36AssignVariableOp+assignvariableop_36_separable_conv2d_9_biasIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37▒
AssignVariableOp_37AssignVariableOp8assignvariableop_37_separable_conv2d_10_depthwise_kernelIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp8assignvariableop_38_separable_conv2d_10_pointwise_kernelIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39е
AssignVariableOp_39AssignVariableOp,assignvariableop_39_separable_conv2d_10_biasIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp8assignvariableop_40_separable_conv2d_11_depthwise_kernelIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41▒
AssignVariableOp_41AssignVariableOp8assignvariableop_41_separable_conv2d_11_pointwise_kernelIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42е
AssignVariableOp_42AssignVariableOp,assignvariableop_42_separable_conv2d_11_biasIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43▒
AssignVariableOp_43AssignVariableOp8assignvariableop_43_separable_conv2d_12_depthwise_kernelIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp8assignvariableop_44_separable_conv2d_12_pointwise_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45е
AssignVariableOp_45AssignVariableOp,assignvariableop_45_separable_conv2d_12_biasIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp8assignvariableop_46_separable_conv2d_13_depthwise_kernelIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47▒
AssignVariableOp_47AssignVariableOp8assignvariableop_47_separable_conv2d_13_pointwise_kernelIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48е
AssignVariableOp_48AssignVariableOp,assignvariableop_48_separable_conv2d_13_biasIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49▒
AssignVariableOp_49AssignVariableOp8assignvariableop_49_separable_conv2d_14_depthwise_kernelIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50▒
AssignVariableOp_50AssignVariableOp8assignvariableop_50_separable_conv2d_14_pointwise_kernelIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51е
AssignVariableOp_51AssignVariableOp,assignvariableop_51_separable_conv2d_14_biasIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52▒
AssignVariableOp_52AssignVariableOp8assignvariableop_52_separable_conv2d_15_depthwise_kernelIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53▒
AssignVariableOp_53AssignVariableOp8assignvariableop_53_separable_conv2d_15_pointwise_kernelIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54е
AssignVariableOp_54AssignVariableOp,assignvariableop_54_separable_conv2d_15_biasIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Ь
AssignVariableOp_55AssignVariableOp#assignvariableop_55_conv2d_2_kernelIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Ъ
AssignVariableOp_56AssignVariableOp!assignvariableop_56_conv2d_2_biasIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57е
AssignVariableOp_57AssignVariableOp,assignvariableop_57_regression_head_1_kernelIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58г
AssignVariableOp_58AssignVariableOp*assignvariableop_58_regression_head_1_biasIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59Т
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60Т
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61Ь
AssignVariableOp_61AssignVariableOp#assignvariableop_61_conv2d_kernel_mIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62Ъ
AssignVariableOp_62AssignVariableOp!assignvariableop_62_conv2d_bias_mIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63░
AssignVariableOp_63AssignVariableOp7assignvariableop_63_separable_conv2d_depthwise_kernel_mIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64░
AssignVariableOp_64AssignVariableOp7assignvariableop_64_separable_conv2d_pointwise_kernel_mIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65д
AssignVariableOp_65AssignVariableOp+assignvariableop_65_separable_conv2d_bias_mIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66▓
AssignVariableOp_66AssignVariableOp9assignvariableop_66_separable_conv2d_1_depthwise_kernel_mIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67▓
AssignVariableOp_67AssignVariableOp9assignvariableop_67_separable_conv2d_1_pointwise_kernel_mIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68ж
AssignVariableOp_68AssignVariableOp-assignvariableop_68_separable_conv2d_1_bias_mIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69Ю
AssignVariableOp_69AssignVariableOp%assignvariableop_69_conv2d_1_kernel_mIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70Ь
AssignVariableOp_70AssignVariableOp#assignvariableop_70_conv2d_1_bias_mIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71▓
AssignVariableOp_71AssignVariableOp9assignvariableop_71_separable_conv2d_2_depthwise_kernel_mIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72▓
AssignVariableOp_72AssignVariableOp9assignvariableop_72_separable_conv2d_2_pointwise_kernel_mIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73ж
AssignVariableOp_73AssignVariableOp-assignvariableop_73_separable_conv2d_2_bias_mIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74▓
AssignVariableOp_74AssignVariableOp9assignvariableop_74_separable_conv2d_3_depthwise_kernel_mIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75▓
AssignVariableOp_75AssignVariableOp9assignvariableop_75_separable_conv2d_3_pointwise_kernel_mIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76ж
AssignVariableOp_76AssignVariableOp-assignvariableop_76_separable_conv2d_3_bias_mIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77▓
AssignVariableOp_77AssignVariableOp9assignvariableop_77_separable_conv2d_4_depthwise_kernel_mIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78▓
AssignVariableOp_78AssignVariableOp9assignvariableop_78_separable_conv2d_4_pointwise_kernel_mIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79ж
AssignVariableOp_79AssignVariableOp-assignvariableop_79_separable_conv2d_4_bias_mIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80▓
AssignVariableOp_80AssignVariableOp9assignvariableop_80_separable_conv2d_5_depthwise_kernel_mIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81▓
AssignVariableOp_81AssignVariableOp9assignvariableop_81_separable_conv2d_5_pointwise_kernel_mIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82ж
AssignVariableOp_82AssignVariableOp-assignvariableop_82_separable_conv2d_5_bias_mIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:2
Identity_83▓
AssignVariableOp_83AssignVariableOp9assignvariableop_83_separable_conv2d_6_depthwise_kernel_mIdentity_83:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_83_
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:2
Identity_84▓
AssignVariableOp_84AssignVariableOp9assignvariableop_84_separable_conv2d_6_pointwise_kernel_mIdentity_84:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_84_
Identity_85IdentityRestoreV2:tensors:85*
T0*
_output_shapes
:2
Identity_85ж
AssignVariableOp_85AssignVariableOp-assignvariableop_85_separable_conv2d_6_bias_mIdentity_85:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_85_
Identity_86IdentityRestoreV2:tensors:86*
T0*
_output_shapes
:2
Identity_86▓
AssignVariableOp_86AssignVariableOp9assignvariableop_86_separable_conv2d_7_depthwise_kernel_mIdentity_86:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_86_
Identity_87IdentityRestoreV2:tensors:87*
T0*
_output_shapes
:2
Identity_87▓
AssignVariableOp_87AssignVariableOp9assignvariableop_87_separable_conv2d_7_pointwise_kernel_mIdentity_87:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_87_
Identity_88IdentityRestoreV2:tensors:88*
T0*
_output_shapes
:2
Identity_88ж
AssignVariableOp_88AssignVariableOp-assignvariableop_88_separable_conv2d_7_bias_mIdentity_88:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_88_
Identity_89IdentityRestoreV2:tensors:89*
T0*
_output_shapes
:2
Identity_89▓
AssignVariableOp_89AssignVariableOp9assignvariableop_89_separable_conv2d_8_depthwise_kernel_mIdentity_89:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_89_
Identity_90IdentityRestoreV2:tensors:90*
T0*
_output_shapes
:2
Identity_90▓
AssignVariableOp_90AssignVariableOp9assignvariableop_90_separable_conv2d_8_pointwise_kernel_mIdentity_90:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_90_
Identity_91IdentityRestoreV2:tensors:91*
T0*
_output_shapes
:2
Identity_91ж
AssignVariableOp_91AssignVariableOp-assignvariableop_91_separable_conv2d_8_bias_mIdentity_91:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_91_
Identity_92IdentityRestoreV2:tensors:92*
T0*
_output_shapes
:2
Identity_92▓
AssignVariableOp_92AssignVariableOp9assignvariableop_92_separable_conv2d_9_depthwise_kernel_mIdentity_92:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_92_
Identity_93IdentityRestoreV2:tensors:93*
T0*
_output_shapes
:2
Identity_93▓
AssignVariableOp_93AssignVariableOp9assignvariableop_93_separable_conv2d_9_pointwise_kernel_mIdentity_93:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_93_
Identity_94IdentityRestoreV2:tensors:94*
T0*
_output_shapes
:2
Identity_94ж
AssignVariableOp_94AssignVariableOp-assignvariableop_94_separable_conv2d_9_bias_mIdentity_94:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_94_
Identity_95IdentityRestoreV2:tensors:95*
T0*
_output_shapes
:2
Identity_95│
AssignVariableOp_95AssignVariableOp:assignvariableop_95_separable_conv2d_10_depthwise_kernel_mIdentity_95:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_95_
Identity_96IdentityRestoreV2:tensors:96*
T0*
_output_shapes
:2
Identity_96│
AssignVariableOp_96AssignVariableOp:assignvariableop_96_separable_conv2d_10_pointwise_kernel_mIdentity_96:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_96_
Identity_97IdentityRestoreV2:tensors:97*
T0*
_output_shapes
:2
Identity_97з
AssignVariableOp_97AssignVariableOp.assignvariableop_97_separable_conv2d_10_bias_mIdentity_97:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_97_
Identity_98IdentityRestoreV2:tensors:98*
T0*
_output_shapes
:2
Identity_98│
AssignVariableOp_98AssignVariableOp:assignvariableop_98_separable_conv2d_11_depthwise_kernel_mIdentity_98:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_98_
Identity_99IdentityRestoreV2:tensors:99*
T0*
_output_shapes
:2
Identity_99│
AssignVariableOp_99AssignVariableOp:assignvariableop_99_separable_conv2d_11_pointwise_kernel_mIdentity_99:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_99b
Identity_100IdentityRestoreV2:tensors:100*
T0*
_output_shapes
:2
Identity_100л
AssignVariableOp_100AssignVariableOp/assignvariableop_100_separable_conv2d_11_bias_mIdentity_100:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_100b
Identity_101IdentityRestoreV2:tensors:101*
T0*
_output_shapes
:2
Identity_101╖
AssignVariableOp_101AssignVariableOp;assignvariableop_101_separable_conv2d_12_depthwise_kernel_mIdentity_101:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_101b
Identity_102IdentityRestoreV2:tensors:102*
T0*
_output_shapes
:2
Identity_102╖
AssignVariableOp_102AssignVariableOp;assignvariableop_102_separable_conv2d_12_pointwise_kernel_mIdentity_102:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_102b
Identity_103IdentityRestoreV2:tensors:103*
T0*
_output_shapes
:2
Identity_103л
AssignVariableOp_103AssignVariableOp/assignvariableop_103_separable_conv2d_12_bias_mIdentity_103:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_103b
Identity_104IdentityRestoreV2:tensors:104*
T0*
_output_shapes
:2
Identity_104╖
AssignVariableOp_104AssignVariableOp;assignvariableop_104_separable_conv2d_13_depthwise_kernel_mIdentity_104:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_104b
Identity_105IdentityRestoreV2:tensors:105*
T0*
_output_shapes
:2
Identity_105╖
AssignVariableOp_105AssignVariableOp;assignvariableop_105_separable_conv2d_13_pointwise_kernel_mIdentity_105:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_105b
Identity_106IdentityRestoreV2:tensors:106*
T0*
_output_shapes
:2
Identity_106л
AssignVariableOp_106AssignVariableOp/assignvariableop_106_separable_conv2d_13_bias_mIdentity_106:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_106b
Identity_107IdentityRestoreV2:tensors:107*
T0*
_output_shapes
:2
Identity_107╖
AssignVariableOp_107AssignVariableOp;assignvariableop_107_separable_conv2d_14_depthwise_kernel_mIdentity_107:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_107b
Identity_108IdentityRestoreV2:tensors:108*
T0*
_output_shapes
:2
Identity_108╖
AssignVariableOp_108AssignVariableOp;assignvariableop_108_separable_conv2d_14_pointwise_kernel_mIdentity_108:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_108b
Identity_109IdentityRestoreV2:tensors:109*
T0*
_output_shapes
:2
Identity_109л
AssignVariableOp_109AssignVariableOp/assignvariableop_109_separable_conv2d_14_bias_mIdentity_109:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_109b
Identity_110IdentityRestoreV2:tensors:110*
T0*
_output_shapes
:2
Identity_110╖
AssignVariableOp_110AssignVariableOp;assignvariableop_110_separable_conv2d_15_depthwise_kernel_mIdentity_110:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_110b
Identity_111IdentityRestoreV2:tensors:111*
T0*
_output_shapes
:2
Identity_111╖
AssignVariableOp_111AssignVariableOp;assignvariableop_111_separable_conv2d_15_pointwise_kernel_mIdentity_111:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_111b
Identity_112IdentityRestoreV2:tensors:112*
T0*
_output_shapes
:2
Identity_112л
AssignVariableOp_112AssignVariableOp/assignvariableop_112_separable_conv2d_15_bias_mIdentity_112:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_112b
Identity_113IdentityRestoreV2:tensors:113*
T0*
_output_shapes
:2
Identity_113в
AssignVariableOp_113AssignVariableOp&assignvariableop_113_conv2d_2_kernel_mIdentity_113:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_113b
Identity_114IdentityRestoreV2:tensors:114*
T0*
_output_shapes
:2
Identity_114а
AssignVariableOp_114AssignVariableOp$assignvariableop_114_conv2d_2_bias_mIdentity_114:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_114b
Identity_115IdentityRestoreV2:tensors:115*
T0*
_output_shapes
:2
Identity_115л
AssignVariableOp_115AssignVariableOp/assignvariableop_115_regression_head_1_kernel_mIdentity_115:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_115b
Identity_116IdentityRestoreV2:tensors:116*
T0*
_output_shapes
:2
Identity_116й
AssignVariableOp_116AssignVariableOp-assignvariableop_116_regression_head_1_bias_mIdentity_116:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_116b
Identity_117IdentityRestoreV2:tensors:117*
T0*
_output_shapes
:2
Identity_117а
AssignVariableOp_117AssignVariableOp$assignvariableop_117_conv2d_kernel_vIdentity_117:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_117b
Identity_118IdentityRestoreV2:tensors:118*
T0*
_output_shapes
:2
Identity_118Ю
AssignVariableOp_118AssignVariableOp"assignvariableop_118_conv2d_bias_vIdentity_118:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_118b
Identity_119IdentityRestoreV2:tensors:119*
T0*
_output_shapes
:2
Identity_119┤
AssignVariableOp_119AssignVariableOp8assignvariableop_119_separable_conv2d_depthwise_kernel_vIdentity_119:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_119b
Identity_120IdentityRestoreV2:tensors:120*
T0*
_output_shapes
:2
Identity_120┤
AssignVariableOp_120AssignVariableOp8assignvariableop_120_separable_conv2d_pointwise_kernel_vIdentity_120:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_120b
Identity_121IdentityRestoreV2:tensors:121*
T0*
_output_shapes
:2
Identity_121и
AssignVariableOp_121AssignVariableOp,assignvariableop_121_separable_conv2d_bias_vIdentity_121:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_121b
Identity_122IdentityRestoreV2:tensors:122*
T0*
_output_shapes
:2
Identity_122╢
AssignVariableOp_122AssignVariableOp:assignvariableop_122_separable_conv2d_1_depthwise_kernel_vIdentity_122:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_122b
Identity_123IdentityRestoreV2:tensors:123*
T0*
_output_shapes
:2
Identity_123╢
AssignVariableOp_123AssignVariableOp:assignvariableop_123_separable_conv2d_1_pointwise_kernel_vIdentity_123:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_123b
Identity_124IdentityRestoreV2:tensors:124*
T0*
_output_shapes
:2
Identity_124к
AssignVariableOp_124AssignVariableOp.assignvariableop_124_separable_conv2d_1_bias_vIdentity_124:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_124b
Identity_125IdentityRestoreV2:tensors:125*
T0*
_output_shapes
:2
Identity_125в
AssignVariableOp_125AssignVariableOp&assignvariableop_125_conv2d_1_kernel_vIdentity_125:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_125b
Identity_126IdentityRestoreV2:tensors:126*
T0*
_output_shapes
:2
Identity_126а
AssignVariableOp_126AssignVariableOp$assignvariableop_126_conv2d_1_bias_vIdentity_126:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_126b
Identity_127IdentityRestoreV2:tensors:127*
T0*
_output_shapes
:2
Identity_127╢
AssignVariableOp_127AssignVariableOp:assignvariableop_127_separable_conv2d_2_depthwise_kernel_vIdentity_127:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_127b
Identity_128IdentityRestoreV2:tensors:128*
T0*
_output_shapes
:2
Identity_128╢
AssignVariableOp_128AssignVariableOp:assignvariableop_128_separable_conv2d_2_pointwise_kernel_vIdentity_128:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_128b
Identity_129IdentityRestoreV2:tensors:129*
T0*
_output_shapes
:2
Identity_129к
AssignVariableOp_129AssignVariableOp.assignvariableop_129_separable_conv2d_2_bias_vIdentity_129:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_129b
Identity_130IdentityRestoreV2:tensors:130*
T0*
_output_shapes
:2
Identity_130╢
AssignVariableOp_130AssignVariableOp:assignvariableop_130_separable_conv2d_3_depthwise_kernel_vIdentity_130:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_130b
Identity_131IdentityRestoreV2:tensors:131*
T0*
_output_shapes
:2
Identity_131╢
AssignVariableOp_131AssignVariableOp:assignvariableop_131_separable_conv2d_3_pointwise_kernel_vIdentity_131:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_131b
Identity_132IdentityRestoreV2:tensors:132*
T0*
_output_shapes
:2
Identity_132к
AssignVariableOp_132AssignVariableOp.assignvariableop_132_separable_conv2d_3_bias_vIdentity_132:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_132b
Identity_133IdentityRestoreV2:tensors:133*
T0*
_output_shapes
:2
Identity_133╢
AssignVariableOp_133AssignVariableOp:assignvariableop_133_separable_conv2d_4_depthwise_kernel_vIdentity_133:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_133b
Identity_134IdentityRestoreV2:tensors:134*
T0*
_output_shapes
:2
Identity_134╢
AssignVariableOp_134AssignVariableOp:assignvariableop_134_separable_conv2d_4_pointwise_kernel_vIdentity_134:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_134b
Identity_135IdentityRestoreV2:tensors:135*
T0*
_output_shapes
:2
Identity_135к
AssignVariableOp_135AssignVariableOp.assignvariableop_135_separable_conv2d_4_bias_vIdentity_135:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_135b
Identity_136IdentityRestoreV2:tensors:136*
T0*
_output_shapes
:2
Identity_136╢
AssignVariableOp_136AssignVariableOp:assignvariableop_136_separable_conv2d_5_depthwise_kernel_vIdentity_136:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_136b
Identity_137IdentityRestoreV2:tensors:137*
T0*
_output_shapes
:2
Identity_137╢
AssignVariableOp_137AssignVariableOp:assignvariableop_137_separable_conv2d_5_pointwise_kernel_vIdentity_137:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_137b
Identity_138IdentityRestoreV2:tensors:138*
T0*
_output_shapes
:2
Identity_138к
AssignVariableOp_138AssignVariableOp.assignvariableop_138_separable_conv2d_5_bias_vIdentity_138:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_138b
Identity_139IdentityRestoreV2:tensors:139*
T0*
_output_shapes
:2
Identity_139╢
AssignVariableOp_139AssignVariableOp:assignvariableop_139_separable_conv2d_6_depthwise_kernel_vIdentity_139:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_139b
Identity_140IdentityRestoreV2:tensors:140*
T0*
_output_shapes
:2
Identity_140╢
AssignVariableOp_140AssignVariableOp:assignvariableop_140_separable_conv2d_6_pointwise_kernel_vIdentity_140:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_140b
Identity_141IdentityRestoreV2:tensors:141*
T0*
_output_shapes
:2
Identity_141к
AssignVariableOp_141AssignVariableOp.assignvariableop_141_separable_conv2d_6_bias_vIdentity_141:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_141b
Identity_142IdentityRestoreV2:tensors:142*
T0*
_output_shapes
:2
Identity_142╢
AssignVariableOp_142AssignVariableOp:assignvariableop_142_separable_conv2d_7_depthwise_kernel_vIdentity_142:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_142b
Identity_143IdentityRestoreV2:tensors:143*
T0*
_output_shapes
:2
Identity_143╢
AssignVariableOp_143AssignVariableOp:assignvariableop_143_separable_conv2d_7_pointwise_kernel_vIdentity_143:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_143b
Identity_144IdentityRestoreV2:tensors:144*
T0*
_output_shapes
:2
Identity_144к
AssignVariableOp_144AssignVariableOp.assignvariableop_144_separable_conv2d_7_bias_vIdentity_144:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_144b
Identity_145IdentityRestoreV2:tensors:145*
T0*
_output_shapes
:2
Identity_145╢
AssignVariableOp_145AssignVariableOp:assignvariableop_145_separable_conv2d_8_depthwise_kernel_vIdentity_145:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_145b
Identity_146IdentityRestoreV2:tensors:146*
T0*
_output_shapes
:2
Identity_146╢
AssignVariableOp_146AssignVariableOp:assignvariableop_146_separable_conv2d_8_pointwise_kernel_vIdentity_146:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_146b
Identity_147IdentityRestoreV2:tensors:147*
T0*
_output_shapes
:2
Identity_147к
AssignVariableOp_147AssignVariableOp.assignvariableop_147_separable_conv2d_8_bias_vIdentity_147:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_147b
Identity_148IdentityRestoreV2:tensors:148*
T0*
_output_shapes
:2
Identity_148╢
AssignVariableOp_148AssignVariableOp:assignvariableop_148_separable_conv2d_9_depthwise_kernel_vIdentity_148:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_148b
Identity_149IdentityRestoreV2:tensors:149*
T0*
_output_shapes
:2
Identity_149╢
AssignVariableOp_149AssignVariableOp:assignvariableop_149_separable_conv2d_9_pointwise_kernel_vIdentity_149:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_149b
Identity_150IdentityRestoreV2:tensors:150*
T0*
_output_shapes
:2
Identity_150к
AssignVariableOp_150AssignVariableOp.assignvariableop_150_separable_conv2d_9_bias_vIdentity_150:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_150b
Identity_151IdentityRestoreV2:tensors:151*
T0*
_output_shapes
:2
Identity_151╖
AssignVariableOp_151AssignVariableOp;assignvariableop_151_separable_conv2d_10_depthwise_kernel_vIdentity_151:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_151b
Identity_152IdentityRestoreV2:tensors:152*
T0*
_output_shapes
:2
Identity_152╖
AssignVariableOp_152AssignVariableOp;assignvariableop_152_separable_conv2d_10_pointwise_kernel_vIdentity_152:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_152b
Identity_153IdentityRestoreV2:tensors:153*
T0*
_output_shapes
:2
Identity_153л
AssignVariableOp_153AssignVariableOp/assignvariableop_153_separable_conv2d_10_bias_vIdentity_153:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_153b
Identity_154IdentityRestoreV2:tensors:154*
T0*
_output_shapes
:2
Identity_154╖
AssignVariableOp_154AssignVariableOp;assignvariableop_154_separable_conv2d_11_depthwise_kernel_vIdentity_154:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_154b
Identity_155IdentityRestoreV2:tensors:155*
T0*
_output_shapes
:2
Identity_155╖
AssignVariableOp_155AssignVariableOp;assignvariableop_155_separable_conv2d_11_pointwise_kernel_vIdentity_155:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_155b
Identity_156IdentityRestoreV2:tensors:156*
T0*
_output_shapes
:2
Identity_156л
AssignVariableOp_156AssignVariableOp/assignvariableop_156_separable_conv2d_11_bias_vIdentity_156:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_156b
Identity_157IdentityRestoreV2:tensors:157*
T0*
_output_shapes
:2
Identity_157╖
AssignVariableOp_157AssignVariableOp;assignvariableop_157_separable_conv2d_12_depthwise_kernel_vIdentity_157:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_157b
Identity_158IdentityRestoreV2:tensors:158*
T0*
_output_shapes
:2
Identity_158╖
AssignVariableOp_158AssignVariableOp;assignvariableop_158_separable_conv2d_12_pointwise_kernel_vIdentity_158:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_158b
Identity_159IdentityRestoreV2:tensors:159*
T0*
_output_shapes
:2
Identity_159л
AssignVariableOp_159AssignVariableOp/assignvariableop_159_separable_conv2d_12_bias_vIdentity_159:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_159b
Identity_160IdentityRestoreV2:tensors:160*
T0*
_output_shapes
:2
Identity_160╖
AssignVariableOp_160AssignVariableOp;assignvariableop_160_separable_conv2d_13_depthwise_kernel_vIdentity_160:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_160b
Identity_161IdentityRestoreV2:tensors:161*
T0*
_output_shapes
:2
Identity_161╖
AssignVariableOp_161AssignVariableOp;assignvariableop_161_separable_conv2d_13_pointwise_kernel_vIdentity_161:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_161b
Identity_162IdentityRestoreV2:tensors:162*
T0*
_output_shapes
:2
Identity_162л
AssignVariableOp_162AssignVariableOp/assignvariableop_162_separable_conv2d_13_bias_vIdentity_162:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_162b
Identity_163IdentityRestoreV2:tensors:163*
T0*
_output_shapes
:2
Identity_163╖
AssignVariableOp_163AssignVariableOp;assignvariableop_163_separable_conv2d_14_depthwise_kernel_vIdentity_163:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_163b
Identity_164IdentityRestoreV2:tensors:164*
T0*
_output_shapes
:2
Identity_164╖
AssignVariableOp_164AssignVariableOp;assignvariableop_164_separable_conv2d_14_pointwise_kernel_vIdentity_164:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_164b
Identity_165IdentityRestoreV2:tensors:165*
T0*
_output_shapes
:2
Identity_165л
AssignVariableOp_165AssignVariableOp/assignvariableop_165_separable_conv2d_14_bias_vIdentity_165:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_165b
Identity_166IdentityRestoreV2:tensors:166*
T0*
_output_shapes
:2
Identity_166╖
AssignVariableOp_166AssignVariableOp;assignvariableop_166_separable_conv2d_15_depthwise_kernel_vIdentity_166:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_166b
Identity_167IdentityRestoreV2:tensors:167*
T0*
_output_shapes
:2
Identity_167╖
AssignVariableOp_167AssignVariableOp;assignvariableop_167_separable_conv2d_15_pointwise_kernel_vIdentity_167:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_167b
Identity_168IdentityRestoreV2:tensors:168*
T0*
_output_shapes
:2
Identity_168л
AssignVariableOp_168AssignVariableOp/assignvariableop_168_separable_conv2d_15_bias_vIdentity_168:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_168b
Identity_169IdentityRestoreV2:tensors:169*
T0*
_output_shapes
:2
Identity_169в
AssignVariableOp_169AssignVariableOp&assignvariableop_169_conv2d_2_kernel_vIdentity_169:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_169b
Identity_170IdentityRestoreV2:tensors:170*
T0*
_output_shapes
:2
Identity_170а
AssignVariableOp_170AssignVariableOp$assignvariableop_170_conv2d_2_bias_vIdentity_170:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_170b
Identity_171IdentityRestoreV2:tensors:171*
T0*
_output_shapes
:2
Identity_171л
AssignVariableOp_171AssignVariableOp/assignvariableop_171_regression_head_1_kernel_vIdentity_171:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_171b
Identity_172IdentityRestoreV2:tensors:172*
T0*
_output_shapes
:2
Identity_172й
AssignVariableOp_172AssignVariableOp-assignvariableop_172_regression_head_1_bias_vIdentity_172:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_172и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЗ
Identity_173Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_173Х
Identity_174IdentityIdentity_173:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_174"%
identity_174Identity_174:output:0*╦
_input_shapes╣
╢: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_170AssignVariableOp_1702,
AssignVariableOp_171AssignVariableOp_1712,
AssignVariableOp_172AssignVariableOp_1722*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
б
╫
2__inference_separable_conv2d_7_layer_call_fn_79781

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_797722
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
є
х
L__inference_regression_head_1_layer_call_and_return_conditional_losses_80270

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╡
╠
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_79570

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1│
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOp║
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateЎ
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Х
о
-__inference_normalization_layer_call_fn_81387

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_normalization_layer_call_and_return_conditional_losses_800532
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
б
╫
2__inference_separable_conv2d_5_layer_call_fn_79729

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_797202
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╝
╧
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_79928

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2
Seluр
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
╒'
╕
%__inference_model_layer_call_fn_80533
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54#
statefulpartitionedcall_args_55#
statefulpartitionedcall_args_56#
statefulpartitionedcall_args_57#
statefulpartitionedcall_args_58
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54statefulpartitionedcall_args_55statefulpartitionedcall_args_56statefulpartitionedcall_args_57statefulpartitionedcall_args_58*F
Tin?
=2;*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_804722
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         @@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╝
serving_defaultи
C
input_18
serving_default_input_1:0         @@E
regression_head_10
StatefulPartitionedCall:0         tensorflow/serving/predict:╕╨
╥▄
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer-21
layer_with_weights-15
layer-22
layer_with_weights-16
layer-23
layer-24
layer_with_weights-17
layer-25
layer_with_weights-18
layer-26
layer-27
layer_with_weights-19
layer-28
layer-29
layer-30
 layer_with_weights-20
 layer-31
!	optimizer
"regularization_losses
#	variables
$trainable_variables
%	keras_api
&
signatures
▄_default_save_signature
▌__call__
+▐&call_and_return_all_conditional_losses"а╙
_tf_keras_modelЕ╙{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 64, 64, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": -1}, "name": "normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["normalization", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_1", "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["separable_conv2d_1", 0, 0, {}], ["conv2d_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_2", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_3", "inbound_nodes": [[["separable_conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["separable_conv2d_3", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_4", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_5", "inbound_nodes": [[["separable_conv2d_4", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["separable_conv2d_5", 0, 0, {}], ["add_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_6", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_7", "inbound_nodes": [[["separable_conv2d_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["separable_conv2d_7", 0, 0, {}], ["add_2", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_8", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_9", "inbound_nodes": [[["separable_conv2d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["separable_conv2d_9", 0, 0, {}], ["add_3", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_10", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_10", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_11", "inbound_nodes": [[["separable_conv2d_10", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["separable_conv2d_11", 0, 0, {}], ["add_4", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_12", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_13", "inbound_nodes": [[["separable_conv2d_12", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["separable_conv2d_13", 0, 0, {}], ["add_5", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_14", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_15", "inbound_nodes": [[["separable_conv2d_14", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["separable_conv2d_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["max_pooling2d", 0, 0, {}], ["conv2d_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "regression_head_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "regression_head_1", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["regression_head_1", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 64, 64, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": -1}, "name": "normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["normalization", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_1", "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["separable_conv2d_1", 0, 0, {}], ["conv2d_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_2", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_3", "inbound_nodes": [[["separable_conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["separable_conv2d_3", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_4", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_5", "inbound_nodes": [[["separable_conv2d_4", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["separable_conv2d_5", 0, 0, {}], ["add_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_6", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_7", "inbound_nodes": [[["separable_conv2d_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["separable_conv2d_7", 0, 0, {}], ["add_2", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_8", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_9", "inbound_nodes": [[["separable_conv2d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["separable_conv2d_9", 0, 0, {}], ["add_3", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_10", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_10", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_11", "inbound_nodes": [[["separable_conv2d_10", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["separable_conv2d_11", 0, 0, {}], ["add_4", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_12", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_13", "inbound_nodes": [[["separable_conv2d_12", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["separable_conv2d_13", 0, 0, {}], ["add_5", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_14", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_15", "inbound_nodes": [[["separable_conv2d_14", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["separable_conv2d_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["max_pooling2d", 0, 0, {}], ["conv2d_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "regression_head_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "regression_head_1", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["regression_head_1", 0, 0]]}}, "training_config": {"loss": {"regression_head_1": "mean_squared_error"}, "metrics": {"regression_head_1": ["mean_squared_error"]}, "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
н"к
_tf_keras_input_layerК{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 64, 64, 1], "config": {"batch_input_shape": [null, 64, 64, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ъ
'state_variables
(_broadcast_shape
)mean
*variance
	+count
,regularization_losses
-	variables
.trainable_variables
/	keras_api
▀__call__
+р&call_and_return_all_conditional_losses"Л
_tf_keras_layerё{"class_name": "Normalization", "name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": -1}}
п

0kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
с__call__
+т&call_and_return_all_conditional_losses"И
_tf_keras_layerю{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
з
6depthwise_kernel
7pointwise_kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"р	
_tf_keras_layer╞	{"class_name": "SeparableConv2D", "name": "separable_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
м
=depthwise_kernel
>pointwise_kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
ё

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"╩
_tf_keras_layer░{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
Є
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"с
_tf_keras_layer╟{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
м
Ndepthwise_kernel
Opointwise_kernel
Pbias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
м
Udepthwise_kernel
Vpointwise_kernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
Ў
\regularization_losses
]	variables
^trainable_variables
_	keras_api
я__call__
+Ё&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}
м
`depthwise_kernel
apointwise_kernel
bbias
cregularization_losses
d	variables
etrainable_variables
f	keras_api
ё__call__
+Є&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
м
gdepthwise_kernel
hpointwise_kernel
ibias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
Ў
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
ї__call__
+Ў&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}}
м
rdepthwise_kernel
spointwise_kernel
tbias
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
ў__call__
+°&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
м
ydepthwise_kernel
zpointwise_kernel
{bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
∙__call__
+·&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
·
Аregularization_losses
Б	variables
Вtrainable_variables
Г	keras_api
√__call__
+№&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}}
│
Дdepthwise_kernel
Еpointwise_kernel
	Жbias
Зregularization_losses
И	variables
Йtrainable_variables
К	keras_api
¤__call__
+■&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
│
Лdepthwise_kernel
Мpointwise_kernel
	Нbias
Оregularization_losses
П	variables
Рtrainable_variables
С	keras_api
 __call__
+А&call_and_return_all_conditional_losses"х	
_tf_keras_layer╦	{"class_name": "SeparableConv2D", "name": "separable_conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
·
Тregularization_losses
У	variables
Фtrainable_variables
Х	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}}
╡
Цdepthwise_kernel
Чpointwise_kernel
	Шbias
Щregularization_losses
Ъ	variables
Ыtrainable_variables
Ь	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"ч	
_tf_keras_layer═	{"class_name": "SeparableConv2D", "name": "separable_conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_10", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
╡
Эdepthwise_kernel
Юpointwise_kernel
	Яbias
аregularization_losses
б	variables
вtrainable_variables
г	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"ч	
_tf_keras_layer═	{"class_name": "SeparableConv2D", "name": "separable_conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
·
дregularization_losses
е	variables
жtrainable_variables
з	keras_api
З__call__
+И&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}}
╡
иdepthwise_kernel
йpointwise_kernel
	кbias
лregularization_losses
м	variables
нtrainable_variables
о	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"ч	
_tf_keras_layer═	{"class_name": "SeparableConv2D", "name": "separable_conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
╡
пdepthwise_kernel
░pointwise_kernel
	▒bias
▓regularization_losses
│	variables
┤trainable_variables
╡	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"ч	
_tf_keras_layer═	{"class_name": "SeparableConv2D", "name": "separable_conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
·
╢regularization_losses
╖	variables
╕trainable_variables
╣	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_6", "trainable": true, "dtype": "float32"}}
╡
║depthwise_kernel
╗pointwise_kernel
	╝bias
╜regularization_losses
╛	variables
┐trainable_variables
└	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"ч	
_tf_keras_layer═	{"class_name": "SeparableConv2D", "name": "separable_conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
╡
┴depthwise_kernel
┬pointwise_kernel
	├bias
─regularization_losses
┼	variables
╞trainable_variables
╟	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"ч	
_tf_keras_layer═	{"class_name": "SeparableConv2D", "name": "separable_conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
■
╚regularization_losses
╔	variables
╩trainable_variables
╦	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"щ
_tf_keras_layer╧{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
°
╠kernel
	═bias
╬regularization_losses
╧	variables
╨trainable_variables
╤	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
·
╥regularization_losses
╙	variables
╘trainable_variables
╒	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Add", "name": "add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_7", "trainable": true, "dtype": "float32"}}
у
╓regularization_losses
╫	variables
╪trainable_variables
┘	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
П
┌kernel
	█bias
▄regularization_losses
▌	variables
▐trainable_variables
▀	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "Dense", "name": "regression_head_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "regression_head_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
л	0mь1mэ6mю7mя8mЁ=mё>mЄ?mєDmЇEmїNmЎOmўPm°Um∙Vm·Wm√`m№am¤bm■gm hmАimБrmВsmГtmДymЕzmЖ{mЗ	ДmИ	ЕmЙ	ЖmК	ЛmЛ	МmМ	НmН	ЦmО	ЧmП	ШmР	ЭmС	ЮmТ	ЯmУ	иmФ	йmХ	кmЦ	пmЧ	░mШ	▒mЩ	║mЪ	╗mЫ	╝mЬ	┴mЭ	┬mЮ	├mЯ	╠mа	═mб	┌mв	█mг0vд1vе6vж7vз8vи=vй>vк?vлDvмEvнNvоOvпPv░Uv▒Vv▓Wv│`v┤av╡bv╢gv╖hv╕iv╣rv║sv╗tv╝yv╜zv╛{v┐	Дv└	Еv┴	Жv┬	Лv├	Мv─	Нv┼	Цv╞	Чv╟	Шv╚	Эv╔	Юv╩	Яv╦	иv╠	йv═	кv╬	пv╧	░v╨	▒v╤	║v╥	╗v╙	╝v╘	┴v╒	┬v╓	├v╫	╠v╪	═v┘	┌v┌	█v█"
	optimizer
 "
trackable_list_wrapper
К
)0
*1
+2
03
14
65
76
87
=8
>9
?10
D11
E12
N13
O14
P15
U16
V17
W18
`19
a20
b21
g22
h23
i24
r25
s26
t27
y28
z29
{30
Д31
Е32
Ж33
Л34
М35
Н36
Ц37
Ч38
Ш39
Э40
Ю41
Я42
и43
й44
к45
п46
░47
▒48
║49
╗50
╝51
┴52
┬53
├54
╠55
═56
┌57
█58"
trackable_list_wrapper
Є
00
11
62
73
84
=5
>6
?7
D8
E9
N10
O11
P12
U13
V14
W15
`16
a17
b18
g19
h20
i21
r22
s23
t24
y25
z26
{27
Д28
Е29
Ж30
Л31
М32
Н33
Ц34
Ч35
Ш36
Э37
Ю38
Я39
и40
й41
к42
п43
░44
▒45
║46
╗47
╝48
┴49
┬50
├51
╠52
═53
┌54
█55"
trackable_list_wrapper
┐
"regularization_losses
 рlayer_regularization_losses
сlayers
тnon_trainable_variables
#	variables
$trainable_variables
уmetrics
▌__call__
▄_default_save_signature
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
-
Эserving_default"
signature_map
C
)mean
*variance
	+count"
trackable_dict_wrapper
 "
trackable_list_wrapper
:2normalization/mean
": 2normalization/variance
: 2normalization/count
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
б
,regularization_losses
 фlayer_regularization_losses
хlayers
цnon_trainable_variables
-	variables
.trainable_variables
чmetrics
▀__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
б
2regularization_losses
 шlayer_regularization_losses
щlayers
ъnon_trainable_variables
3	variables
4trainable_variables
ыmetrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
;:9@2!separable_conv2d/depthwise_kernel
<::@А2!separable_conv2d/pointwise_kernel
$:"А2separable_conv2d/bias
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
б
9regularization_losses
 ьlayer_regularization_losses
эlayers
юnon_trainable_variables
:	variables
;trainable_variables
яmetrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_1/depthwise_kernel
?:=АА2#separable_conv2d_1/pointwise_kernel
&:$А2separable_conv2d_1/bias
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
б
@regularization_losses
 Ёlayer_regularization_losses
ёlayers
Єnon_trainable_variables
A	variables
Btrainable_variables
єmetrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
*:(@А2conv2d_1/kernel
:А2conv2d_1/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
б
Fregularization_losses
 Їlayer_regularization_losses
їlayers
Ўnon_trainable_variables
G	variables
Htrainable_variables
ўmetrics
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Jregularization_losses
 °layer_regularization_losses
∙layers
·non_trainable_variables
K	variables
Ltrainable_variables
√metrics
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_2/depthwise_kernel
?:=АА2#separable_conv2d_2/pointwise_kernel
&:$А2separable_conv2d_2/bias
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
б
Qregularization_losses
 №layer_regularization_losses
¤layers
■non_trainable_variables
R	variables
Strainable_variables
 metrics
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_3/depthwise_kernel
?:=АА2#separable_conv2d_3/pointwise_kernel
&:$А2separable_conv2d_3/bias
 "
trackable_list_wrapper
5
U0
V1
W2"
trackable_list_wrapper
5
U0
V1
W2"
trackable_list_wrapper
б
Xregularization_losses
 Аlayer_regularization_losses
Бlayers
Вnon_trainable_variables
Y	variables
Ztrainable_variables
Гmetrics
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
\regularization_losses
 Дlayer_regularization_losses
Еlayers
Жnon_trainable_variables
]	variables
^trainable_variables
Зmetrics
я__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_4/depthwise_kernel
?:=АА2#separable_conv2d_4/pointwise_kernel
&:$А2separable_conv2d_4/bias
 "
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
б
cregularization_losses
 Иlayer_regularization_losses
Йlayers
Кnon_trainable_variables
d	variables
etrainable_variables
Лmetrics
ё__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_5/depthwise_kernel
?:=АА2#separable_conv2d_5/pointwise_kernel
&:$А2separable_conv2d_5/bias
 "
trackable_list_wrapper
5
g0
h1
i2"
trackable_list_wrapper
5
g0
h1
i2"
trackable_list_wrapper
б
jregularization_losses
 Мlayer_regularization_losses
Нlayers
Оnon_trainable_variables
k	variables
ltrainable_variables
Пmetrics
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
nregularization_losses
 Рlayer_regularization_losses
Сlayers
Тnon_trainable_variables
o	variables
ptrainable_variables
Уmetrics
ї__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_6/depthwise_kernel
?:=АА2#separable_conv2d_6/pointwise_kernel
&:$А2separable_conv2d_6/bias
 "
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
б
uregularization_losses
 Фlayer_regularization_losses
Хlayers
Цnon_trainable_variables
v	variables
wtrainable_variables
Чmetrics
ў__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_7/depthwise_kernel
?:=АА2#separable_conv2d_7/pointwise_kernel
&:$А2separable_conv2d_7/bias
 "
trackable_list_wrapper
5
y0
z1
{2"
trackable_list_wrapper
5
y0
z1
{2"
trackable_list_wrapper
б
|regularization_losses
 Шlayer_regularization_losses
Щlayers
Ъnon_trainable_variables
}	variables
~trainable_variables
Ыmetrics
∙__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
Аregularization_losses
 Ьlayer_regularization_losses
Эlayers
Юnon_trainable_variables
Б	variables
Вtrainable_variables
Яmetrics
√__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_8/depthwise_kernel
?:=АА2#separable_conv2d_8/pointwise_kernel
&:$А2separable_conv2d_8/bias
 "
trackable_list_wrapper
8
Д0
Е1
Ж2"
trackable_list_wrapper
8
Д0
Е1
Ж2"
trackable_list_wrapper
д
Зregularization_losses
 аlayer_regularization_losses
бlayers
вnon_trainable_variables
И	variables
Йtrainable_variables
гmetrics
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_9/depthwise_kernel
?:=АА2#separable_conv2d_9/pointwise_kernel
&:$А2separable_conv2d_9/bias
 "
trackable_list_wrapper
8
Л0
М1
Н2"
trackable_list_wrapper
8
Л0
М1
Н2"
trackable_list_wrapper
д
Оregularization_losses
 дlayer_regularization_losses
еlayers
жnon_trainable_variables
П	variables
Рtrainable_variables
зmetrics
 __call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
Тregularization_losses
 иlayer_regularization_losses
йlayers
кnon_trainable_variables
У	variables
Фtrainable_variables
лmetrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_10/depthwise_kernel
@:>АА2$separable_conv2d_10/pointwise_kernel
':%А2separable_conv2d_10/bias
 "
trackable_list_wrapper
8
Ц0
Ч1
Ш2"
trackable_list_wrapper
8
Ц0
Ч1
Ш2"
trackable_list_wrapper
д
Щregularization_losses
 мlayer_regularization_losses
нlayers
оnon_trainable_variables
Ъ	variables
Ыtrainable_variables
пmetrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_11/depthwise_kernel
@:>АА2$separable_conv2d_11/pointwise_kernel
':%А2separable_conv2d_11/bias
 "
trackable_list_wrapper
8
Э0
Ю1
Я2"
trackable_list_wrapper
8
Э0
Ю1
Я2"
trackable_list_wrapper
д
аregularization_losses
 ░layer_regularization_losses
▒layers
▓non_trainable_variables
б	variables
вtrainable_variables
│metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
дregularization_losses
 ┤layer_regularization_losses
╡layers
╢non_trainable_variables
е	variables
жtrainable_variables
╖metrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_12/depthwise_kernel
@:>АА2$separable_conv2d_12/pointwise_kernel
':%А2separable_conv2d_12/bias
 "
trackable_list_wrapper
8
и0
й1
к2"
trackable_list_wrapper
8
и0
й1
к2"
trackable_list_wrapper
д
лregularization_losses
 ╕layer_regularization_losses
╣layers
║non_trainable_variables
м	variables
нtrainable_variables
╗metrics
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_13/depthwise_kernel
@:>АА2$separable_conv2d_13/pointwise_kernel
':%А2separable_conv2d_13/bias
 "
trackable_list_wrapper
8
п0
░1
▒2"
trackable_list_wrapper
8
п0
░1
▒2"
trackable_list_wrapper
д
▓regularization_losses
 ╝layer_regularization_losses
╜layers
╛non_trainable_variables
│	variables
┤trainable_variables
┐metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
╢regularization_losses
 └layer_regularization_losses
┴layers
┬non_trainable_variables
╖	variables
╕trainable_variables
├metrics
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_14/depthwise_kernel
@:>АА2$separable_conv2d_14/pointwise_kernel
':%А2separable_conv2d_14/bias
 "
trackable_list_wrapper
8
║0
╗1
╝2"
trackable_list_wrapper
8
║0
╗1
╝2"
trackable_list_wrapper
д
╜regularization_losses
 ─layer_regularization_losses
┼layers
╞non_trainable_variables
╛	variables
┐trainable_variables
╟metrics
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_15/depthwise_kernel
@:>АА2$separable_conv2d_15/pointwise_kernel
':%А2separable_conv2d_15/bias
 "
trackable_list_wrapper
8
┴0
┬1
├2"
trackable_list_wrapper
8
┴0
┬1
├2"
trackable_list_wrapper
д
─regularization_losses
 ╚layer_regularization_losses
╔layers
╩non_trainable_variables
┼	variables
╞trainable_variables
╦metrics
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
╚regularization_losses
 ╠layer_regularization_losses
═layers
╬non_trainable_variables
╔	variables
╩trainable_variables
╧metrics
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
+:)АА2conv2d_2/kernel
:А2conv2d_2/bias
 "
trackable_list_wrapper
0
╠0
═1"
trackable_list_wrapper
0
╠0
═1"
trackable_list_wrapper
д
╬regularization_losses
 ╨layer_regularization_losses
╤layers
╥non_trainable_variables
╧	variables
╨trainable_variables
╙metrics
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
╥regularization_losses
 ╘layer_regularization_losses
╒layers
╓non_trainable_variables
╙	variables
╘trainable_variables
╫metrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
╓regularization_losses
 ╪layer_regularization_losses
┘layers
┌non_trainable_variables
╫	variables
╪trainable_variables
█metrics
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
+:)	А2regression_head_1/kernel
$:"2regression_head_1/bias
 "
trackable_list_wrapper
0
┌0
█1"
trackable_list_wrapper
0
┌0
█1"
trackable_list_wrapper
д
▄regularization_losses
 ▄layer_regularization_losses
▌layers
▐non_trainable_variables
▌	variables
▐trainable_variables
▀metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
(
р0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╖

сtotal

тcount
у
_fn_kwargs
фregularization_losses
х	variables
цtrainable_variables
ч	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"∙
_tf_keras_layer▀{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
с0
т1"
trackable_list_wrapper
 "
trackable_list_wrapper
д
фregularization_losses
 шlayer_regularization_losses
щlayers
ъnon_trainable_variables
х	variables
цtrainable_variables
ыmetrics
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
с0
т1"
trackable_list_wrapper
 "
trackable_list_wrapper
':%@2conv2d/kernel/m
:@2conv2d/bias/m
;:9@2#separable_conv2d/depthwise_kernel/m
<::@А2#separable_conv2d/pointwise_kernel/m
$:"А2separable_conv2d/bias/m
>:<А2%separable_conv2d_1/depthwise_kernel/m
?:=АА2%separable_conv2d_1/pointwise_kernel/m
&:$А2separable_conv2d_1/bias/m
*:(@А2conv2d_1/kernel/m
:А2conv2d_1/bias/m
>:<А2%separable_conv2d_2/depthwise_kernel/m
?:=АА2%separable_conv2d_2/pointwise_kernel/m
&:$А2separable_conv2d_2/bias/m
>:<А2%separable_conv2d_3/depthwise_kernel/m
?:=АА2%separable_conv2d_3/pointwise_kernel/m
&:$А2separable_conv2d_3/bias/m
>:<А2%separable_conv2d_4/depthwise_kernel/m
?:=АА2%separable_conv2d_4/pointwise_kernel/m
&:$А2separable_conv2d_4/bias/m
>:<А2%separable_conv2d_5/depthwise_kernel/m
?:=АА2%separable_conv2d_5/pointwise_kernel/m
&:$А2separable_conv2d_5/bias/m
>:<А2%separable_conv2d_6/depthwise_kernel/m
?:=АА2%separable_conv2d_6/pointwise_kernel/m
&:$А2separable_conv2d_6/bias/m
>:<А2%separable_conv2d_7/depthwise_kernel/m
?:=АА2%separable_conv2d_7/pointwise_kernel/m
&:$А2separable_conv2d_7/bias/m
>:<А2%separable_conv2d_8/depthwise_kernel/m
?:=АА2%separable_conv2d_8/pointwise_kernel/m
&:$А2separable_conv2d_8/bias/m
>:<А2%separable_conv2d_9/depthwise_kernel/m
?:=АА2%separable_conv2d_9/pointwise_kernel/m
&:$А2separable_conv2d_9/bias/m
?:=А2&separable_conv2d_10/depthwise_kernel/m
@:>АА2&separable_conv2d_10/pointwise_kernel/m
':%А2separable_conv2d_10/bias/m
?:=А2&separable_conv2d_11/depthwise_kernel/m
@:>АА2&separable_conv2d_11/pointwise_kernel/m
':%А2separable_conv2d_11/bias/m
?:=А2&separable_conv2d_12/depthwise_kernel/m
@:>АА2&separable_conv2d_12/pointwise_kernel/m
':%А2separable_conv2d_12/bias/m
?:=А2&separable_conv2d_13/depthwise_kernel/m
@:>АА2&separable_conv2d_13/pointwise_kernel/m
':%А2separable_conv2d_13/bias/m
?:=А2&separable_conv2d_14/depthwise_kernel/m
@:>АА2&separable_conv2d_14/pointwise_kernel/m
':%А2separable_conv2d_14/bias/m
?:=А2&separable_conv2d_15/depthwise_kernel/m
@:>АА2&separable_conv2d_15/pointwise_kernel/m
':%А2separable_conv2d_15/bias/m
+:)АА2conv2d_2/kernel/m
:А2conv2d_2/bias/m
+:)	А2regression_head_1/kernel/m
$:"2regression_head_1/bias/m
':%@2conv2d/kernel/v
:@2conv2d/bias/v
;:9@2#separable_conv2d/depthwise_kernel/v
<::@А2#separable_conv2d/pointwise_kernel/v
$:"А2separable_conv2d/bias/v
>:<А2%separable_conv2d_1/depthwise_kernel/v
?:=АА2%separable_conv2d_1/pointwise_kernel/v
&:$А2separable_conv2d_1/bias/v
*:(@А2conv2d_1/kernel/v
:А2conv2d_1/bias/v
>:<А2%separable_conv2d_2/depthwise_kernel/v
?:=АА2%separable_conv2d_2/pointwise_kernel/v
&:$А2separable_conv2d_2/bias/v
>:<А2%separable_conv2d_3/depthwise_kernel/v
?:=АА2%separable_conv2d_3/pointwise_kernel/v
&:$А2separable_conv2d_3/bias/v
>:<А2%separable_conv2d_4/depthwise_kernel/v
?:=АА2%separable_conv2d_4/pointwise_kernel/v
&:$А2separable_conv2d_4/bias/v
>:<А2%separable_conv2d_5/depthwise_kernel/v
?:=АА2%separable_conv2d_5/pointwise_kernel/v
&:$А2separable_conv2d_5/bias/v
>:<А2%separable_conv2d_6/depthwise_kernel/v
?:=АА2%separable_conv2d_6/pointwise_kernel/v
&:$А2separable_conv2d_6/bias/v
>:<А2%separable_conv2d_7/depthwise_kernel/v
?:=АА2%separable_conv2d_7/pointwise_kernel/v
&:$А2separable_conv2d_7/bias/v
>:<А2%separable_conv2d_8/depthwise_kernel/v
?:=АА2%separable_conv2d_8/pointwise_kernel/v
&:$А2separable_conv2d_8/bias/v
>:<А2%separable_conv2d_9/depthwise_kernel/v
?:=АА2%separable_conv2d_9/pointwise_kernel/v
&:$А2separable_conv2d_9/bias/v
?:=А2&separable_conv2d_10/depthwise_kernel/v
@:>АА2&separable_conv2d_10/pointwise_kernel/v
':%А2separable_conv2d_10/bias/v
?:=А2&separable_conv2d_11/depthwise_kernel/v
@:>АА2&separable_conv2d_11/pointwise_kernel/v
':%А2separable_conv2d_11/bias/v
?:=А2&separable_conv2d_12/depthwise_kernel/v
@:>АА2&separable_conv2d_12/pointwise_kernel/v
':%А2separable_conv2d_12/bias/v
?:=А2&separable_conv2d_13/depthwise_kernel/v
@:>АА2&separable_conv2d_13/pointwise_kernel/v
':%А2separable_conv2d_13/bias/v
?:=А2&separable_conv2d_14/depthwise_kernel/v
@:>АА2&separable_conv2d_14/pointwise_kernel/v
':%А2separable_conv2d_14/bias/v
?:=А2&separable_conv2d_15/depthwise_kernel/v
@:>АА2&separable_conv2d_15/pointwise_kernel/v
':%А2separable_conv2d_15/bias/v
+:)АА2conv2d_2/kernel/v
:А2conv2d_2/bias/v
+:)	А2regression_head_1/kernel/v
$:"2regression_head_1/bias/v
ц2у
 __inference__wrapped_model_79532╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         @@
т2▀
%__inference_model_layer_call_fn_80689
%__inference_model_layer_call_fn_81302
%__inference_model_layer_call_fn_81365
%__inference_model_layer_call_fn_80533└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
@__inference_model_layer_call_and_return_conditional_losses_80996
@__inference_model_layer_call_and_return_conditional_losses_80283
@__inference_model_layer_call_and_return_conditional_losses_80376
@__inference_model_layer_call_and_return_conditional_losses_81239└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╫2╘
-__inference_normalization_layer_call_fn_81387в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_normalization_layer_call_and_return_conditional_losses_81380в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Е2В
&__inference_conv2d_layer_call_fn_79553╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
а2Э
A__inference_conv2d_layer_call_and_return_conditional_losses_79545╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
П2М
0__inference_separable_conv2d_layer_call_fn_79579╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
к2з
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_79570╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
Т2П
2__inference_separable_conv2d_1_layer_call_fn_79605╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_79596╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
З2Д
(__inference_conv2d_1_layer_call_fn_79625╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
в2Я
C__inference_conv2d_1_layer_call_and_return_conditional_losses_79617╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
═2╩
#__inference_add_layer_call_fn_81399в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ш2х
>__inference_add_layer_call_and_return_conditional_losses_81393в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
2__inference_separable_conv2d_2_layer_call_fn_79651╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_79642╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Т2П
2__inference_separable_conv2d_3_layer_call_fn_79677╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_79668╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_1_layer_call_fn_81411в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_1_layer_call_and_return_conditional_losses_81405в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
2__inference_separable_conv2d_4_layer_call_fn_79703╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_79694╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Т2П
2__inference_separable_conv2d_5_layer_call_fn_79729╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_79720╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_2_layer_call_fn_81423в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_2_layer_call_and_return_conditional_losses_81417в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
2__inference_separable_conv2d_6_layer_call_fn_79755╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_79746╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Т2П
2__inference_separable_conv2d_7_layer_call_fn_79781╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_79772╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_3_layer_call_fn_81435в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_3_layer_call_and_return_conditional_losses_81429в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
2__inference_separable_conv2d_8_layer_call_fn_79807╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_79798╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Т2П
2__inference_separable_conv2d_9_layer_call_fn_79833╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
н2к
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_79824╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_4_layer_call_fn_81447в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_4_layer_call_and_return_conditional_losses_81441в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
У2Р
3__inference_separable_conv2d_10_layer_call_fn_79859╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
о2л
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_79850╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
У2Р
3__inference_separable_conv2d_11_layer_call_fn_79885╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
о2л
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_79876╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_5_layer_call_fn_81459в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_5_layer_call_and_return_conditional_losses_81453в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
У2Р
3__inference_separable_conv2d_12_layer_call_fn_79911╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
о2л
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_79902╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
У2Р
3__inference_separable_conv2d_13_layer_call_fn_79937╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
о2л
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_79928╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_6_layer_call_fn_81471в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_6_layer_call_and_return_conditional_losses_81465в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
У2Р
3__inference_separable_conv2d_14_layer_call_fn_79963╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
о2л
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_79954╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
У2Р
3__inference_separable_conv2d_15_layer_call_fn_79989╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
о2л
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_79980╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Х2Т
-__inference_max_pooling2d_layer_call_fn_80001р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
░2н
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_79995р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
И2Е
(__inference_conv2d_2_layer_call_fn_80021╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
г2а
C__inference_conv2d_2_layer_call_and_return_conditional_losses_80013╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╧2╠
%__inference_add_7_layer_call_fn_81483в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_add_7_layer_call_and_return_conditional_losses_81477в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
а2Э
8__inference_global_average_pooling2d_layer_call_fn_80034р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╗2╕
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_80028р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
█2╪
1__inference_regression_head_1_layer_call_fn_81500в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_regression_head_1_layer_call_and_return_conditional_losses_81493в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
2B0
#__inference_signature_wrapper_80753input_1
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 ■
 __inference__wrapped_model_79532┘V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█8в5
.в+
)К&
input_1         @@
к "EкB
@
regression_head_1+К(
regression_head_1         у
@__inference_add_1_layer_call_and_return_conditional_losses_81405Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╗
%__inference_add_1_layer_call_fn_81411Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           Ау
@__inference_add_2_layer_call_and_return_conditional_losses_81417Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╗
%__inference_add_2_layer_call_fn_81423Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           Ау
@__inference_add_3_layer_call_and_return_conditional_losses_81429Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╗
%__inference_add_3_layer_call_fn_81435Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           Ау
@__inference_add_4_layer_call_and_return_conditional_losses_81441Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╗
%__inference_add_4_layer_call_fn_81447Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           Ау
@__inference_add_5_layer_call_and_return_conditional_losses_81453Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╗
%__inference_add_5_layer_call_fn_81459Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           Ау
@__inference_add_6_layer_call_and_return_conditional_losses_81465Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╗
%__inference_add_6_layer_call_fn_81471Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           Ау
@__inference_add_7_layer_call_and_return_conditional_losses_81477Юlвi
bв_
]ЪZ
+К(
inputs/0         А
+К(
inputs/1         А
к ".в+
$К!
0         А
Ъ ╗
%__inference_add_7_layer_call_fn_81483Сlвi
bв_
]ЪZ
+К(
inputs/0         А
+К(
inputs/1         А
к "!К         Ас
>__inference_add_layer_call_and_return_conditional_losses_81393Юlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к ".в+
$К!
0           А
Ъ ╣
#__inference_add_layer_call_fn_81399Сlвi
bв_
]ЪZ
+К(
inputs/0           А
+К(
inputs/1           А
к "!К           А┘
C__inference_conv2d_1_layer_call_and_return_conditional_losses_79617СDEIвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ▒
(__inference_conv2d_1_layer_call_fn_79625ДDEIвF
?в<
:К7
inputs+                           @
к "3К0,                           А▄
C__inference_conv2d_2_layer_call_and_return_conditional_losses_80013Ф╠═JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┤
(__inference_conv2d_2_layer_call_fn_80021З╠═JвG
@в=
;К8
inputs,                           А
к "3К0,                           А╓
A__inference_conv2d_layer_call_and_return_conditional_losses_79545Р01IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           @
Ъ о
&__inference_conv2d_layer_call_fn_79553Г01IвF
?в<
:К7
inputs+                           
к "2К/+                           @▄
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_80028ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ │
8__inference_global_average_pooling2d_layer_call_fn_80034wRвO
HвE
CК@
inputs4                                    
к "!К                  ы
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_79995ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ├
-__inference_max_pooling2d_layer_call_fn_80001СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ж
@__inference_model_layer_call_and_return_conditional_losses_80283┴V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█@в=
6в3
)К&
input_1         @@
p

 
к "%в"
К
0         
Ъ Ж
@__inference_model_layer_call_and_return_conditional_losses_80376┴V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█@в=
6в3
)К&
input_1         @@
p 

 
к "%в"
К
0         
Ъ Е
@__inference_model_layer_call_and_return_conditional_losses_80996└V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█?в<
5в2
(К%
inputs         @@
p

 
к "%в"
К
0         
Ъ Е
@__inference_model_layer_call_and_return_conditional_losses_81239└V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█?в<
5в2
(К%
inputs         @@
p 

 
к "%в"
К
0         
Ъ ▐
%__inference_model_layer_call_fn_80533┤V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█@в=
6в3
)К&
input_1         @@
p

 
к "К         ▐
%__inference_model_layer_call_fn_80689┤V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█@в=
6в3
)К&
input_1         @@
p 

 
к "К         ▌
%__inference_model_layer_call_fn_81302│V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█?в<
5в2
(К%
inputs         @@
p

 
к "К         ▌
%__inference_model_layer_call_fn_81365│V)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█?в<
5в2
(К%
inputs         @@
p 

 
к "К         ╕
H__inference_normalization_layer_call_and_return_conditional_losses_81380l)*7в4
-в*
(К%
inputs         @@
к "-в*
#К 
0         @@
Ъ Р
-__inference_normalization_layer_call_fn_81387_)*7в4
-в*
(К%
inputs         @@
к " К         @@п
L__inference_regression_head_1_layer_call_and_return_conditional_losses_81493_┌█0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ З
1__inference_regression_head_1_layer_call_fn_81500R┌█0в-
&в#
!К
inputs         А
к "К         щ
N__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_79850ЦЦЧШJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┴
3__inference_separable_conv2d_10_layer_call_fn_79859ЙЦЧШJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ащ
N__inference_separable_conv2d_11_layer_call_and_return_conditional_losses_79876ЦЭЮЯJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┴
3__inference_separable_conv2d_11_layer_call_fn_79885ЙЭЮЯJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ащ
N__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_79902ЦийкJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┴
3__inference_separable_conv2d_12_layer_call_fn_79911ЙийкJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ащ
N__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_79928Цп░▒JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┴
3__inference_separable_conv2d_13_layer_call_fn_79937Йп░▒JвG
@в=
;К8
inputs,                           А
к "3К0,                           Ащ
N__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_79954Ц║╗╝JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┴
3__inference_separable_conv2d_14_layer_call_fn_79963Й║╗╝JвG
@в=
;К8
inputs,                           А
к "3К0,                           Ащ
N__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_79980Ц┴┬├JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┴
3__inference_separable_conv2d_15_layer_call_fn_79989Й┴┬├JвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_79596У=>?JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_1_layer_call_fn_79605Ж=>?JвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_79642УNOPJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_2_layer_call_fn_79651ЖNOPJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_79668УUVWJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_3_layer_call_fn_79677ЖUVWJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_79694У`abJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_4_layer_call_fn_79703Ж`abJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_79720УghiJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_5_layer_call_fn_79729ЖghiJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_79746УrstJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_6_layer_call_fn_79755ЖrstJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ах
M__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_79772Уyz{JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
2__inference_separable_conv2d_7_layer_call_fn_79781Жyz{JвG
@в=
;К8
inputs,                           А
к "3К0,                           Аш
M__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_79798ЦДЕЖJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ └
2__inference_separable_conv2d_8_layer_call_fn_79807ЙДЕЖJвG
@в=
;К8
inputs,                           А
к "3К0,                           Аш
M__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_79824ЦЛМНJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ └
2__inference_separable_conv2d_9_layer_call_fn_79833ЙЛМНJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ат
K__inference_separable_conv2d_layer_call_and_return_conditional_losses_79570Т678IвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ║
0__inference_separable_conv2d_layer_call_fn_79579Е678IвF
?в<
:К7
inputs+                           @
к "3К0,                           АМ
#__inference_signature_wrapper_80753фV)*01678=>?DENOPUVW`abghirstyz{ДЕЖЛМНЦЧШЭЮЯийкп░▒║╗╝┴┬├╠═┌█Cв@
в 
9к6
4
input_1)К&
input_1         @@"EкB
@
regression_head_1+К(
regression_head_1         