-define (NO_TRANSPOSE, 0).
-define (TRANSPOSE, 1).
-define (CONJUGATE_TRANSPOSE, 2).

-define (ROW_MAJOR, 0).
-define (COLUMN_MAJOR, 1).

-type int_vector() :: [integer()].
-type float_vector() :: [float()].
-type int_matrix() :: [[integer()]].
-type float_matrix() :: [[float()]].
-type matrix() :: int_matrix() | float_matrix().
-type matrix_rows() :: integer().
-type matrix_columns() :: integer().
-type storage_type() :: vector | matrix.
-type data_type() :: integer | float | string.
-type orientation() :: row_major | column_major.


-record(pc_buffer, {type         :: storage_type(),
					data_type    :: data_type(),
					orientation  :: orientation(),
                    ref          :: term()}).

-record(pc_context, {ref :: term()}).

-type buffer() :: #pc_buffer{}.
-type float_matrix_buffer() :: #pc_buffer{type::matrix, data_type::float}.
-type float_vector_buffer() :: #pc_buffer{type::vector, data_type::float}.

-type context() :: #pc_context{}.

