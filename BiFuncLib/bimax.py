def cbimax(logical_matrix, minr=2, minc=2, number=100):
    rows = len(logical_matrix)
    cols = len(logical_matrix[0]) if rows else 0
    minr = max(1, minr)
    minc = max(1, minc)
    bits_per_int = 32
    no_ints = (cols // bits_per_int) + (1 if cols % bits_per_int else 0)
    row_objs = []
    for i, row in enumerate(logical_matrix):
        vec = [0] * no_ints
        for j, val in enumerate(row):
            if val:
                vec_idx = j // bits_per_int
                bit_pos = j % bits_per_int
                vec[vec_idx] |= (1 << bit_pos)
        row_objs.append({'orig': i, 'vec': vec})
    results = []
    stop_flag = [False]
    def count_bits(vec):
        count = 0
        for v in vec:
            while v:
                count += v & 1
                v //= 2
        return count
    def intersect_vec(a, b):
        return [x & y for x, y in zip(a, b)]
    def union_vec(a, b):
        return [x | y for x, y in zip(a, b)]
    def get_columns(vec):
        columns = set()
        for vec_idx, v in enumerate(vec):
            for bit_pos in range(bits_per_int):
                if v & (1 << bit_pos):
                    col = vec_idx * bits_per_int + bit_pos
                    if col < cols:
                        columns.add(col)
        return columns
    def conquer(first, last, considered_vec, mandatory_vec):
        if stop_flag[0] or first > last:
            return
        common_vec = row_objs[first]['vec'][:]
        for i in range(first + 1, last + 1):
            common_vec = intersect_vec(common_vec, row_objs[i]['vec'])
        common_vec = intersect_vec(common_vec, considered_vec)
        common_vec = union_vec(common_vec, mandatory_vec)
        region_rows = last - first + 1
        if region_rows >= minr and count_bits(common_vec) >= minc:
            rows_res = sorted(row_objs[i]['orig'] for i in range(first, last + 1))
            cols_res = sorted(get_columns(common_vec))
            duplicate = False
            for bc in results:
                if bc['rows'] == rows_res and bc['cols'] == cols_res:
                    duplicate = True
                    break
            if not duplicate:
                results.append({'rows': rows_res, 'cols': cols_res})
                if len(results) >= number:
                    stop_flag[0] = True
                    return
        cand = [c for c in get_columns(considered_vec) if c not in get_columns(mandatory_vec)]
        if not cand:
            return
        original_region = row_objs[first:last + 1].copy()
        for col in cand:
            left_count = 0
            for i in range(first, last + 1):
                vec_idx = col // bits_per_int
                bit_pos = col % bits_per_int
                if row_objs[i]['vec'][vec_idx] & (1 << bit_pos):
                    left_count += 1
            total = last - first + 1
            right_count = total - left_count
            if left_count == 0 or left_count == total:
                continue
            vec_idx = col // bits_per_int
            bit_pos = col % bits_per_int
            if left_count >= minr:
                new_mandatory = mandatory_vec.copy()
                new_mandatory[vec_idx] |= (1 << bit_pos)
                store_ptr = first
                for i in range(first, last + 1):
                    if row_objs[i]['vec'][vec_idx] & (1 << bit_pos):
                        if i != store_ptr:
                            row_objs[store_ptr], row_objs[i] = row_objs[i], row_objs[store_ptr]
                        store_ptr += 1
                conquer(first, store_ptr - 1, considered_vec.copy(), new_mandatory)
                if stop_flag[0]:
                    return
            if right_count >= minr:
                new_considered = considered_vec.copy()
                new_considered[vec_idx] &= ~(1 << bit_pos)
                store_ptr = first
                for i in range(first, last + 1):
                    if not (row_objs[i]['vec'][vec_idx] & (1 << bit_pos)):
                        if i != store_ptr:
                            row_objs[store_ptr], row_objs[i] = row_objs[i], row_objs[store_ptr]
                        store_ptr += 1
                conquer(first, store_ptr - 1, new_considered, mandatory_vec.copy())
                if stop_flag[0]:
                    return
            row_objs[first:last + 1] = original_region.copy()
    initial_considered = [0] * no_ints
    for col in range(cols):
        vec_idx = col // bits_per_int
        bit_pos = col % bits_per_int
        initial_considered[vec_idx] |= (1 << bit_pos)
    if rows > 0 and cols > 0:
        conquer(0, rows - 1, initial_considered, [0] * no_ints)
    maximal_results = []
    for bc in results:
        bc_rows = set(bc['rows'])
        bc_cols = set(bc['cols'])
        is_sub = False
        for other in results:
            if bc is other:
                continue
            other_rows = set(other['rows'])
            other_cols = set(other['cols'])
            if bc_rows.issubset(other_rows) and bc_cols.issubset(other_cols):
                if bc_rows != other_rows or bc_cols != other_cols:
                    is_sub = True
                    break
        if not is_sub:
            maximal_results.append(bc)
    final = []
    seen = set()
    for bc in maximal_results:
        key = (tuple(bc['rows']), tuple(bc['cols']))
        if key not in seen:
            seen.add(key)
            final.append(bc)
    final = final[:number]
    return final

def bimaxbiclust(matrix, minr=2, minc=2, number=100):
    raw = cbimax(matrix, minr, minc, number)
    row_size = len(matrix)
    col_size = len(matrix[0]) if row_size else 0
    for bc in raw:
        bc['rows'] = sorted(bc['rows'])
        bc['cols'] = sorted(bc['cols'])
    row_matrix = [[False] * len(raw) for _ in range(row_size)]
    col_matrix = [[False] * col_size for _ in range(len(raw))]
    for bc_idx, bc in enumerate(raw):
        for r in bc['rows']:
            row_matrix[r][bc_idx] = True
        for c in bc['cols']:
            col_matrix[bc_idx][c] = True
    return {
        'Parameters': {
            'Algorithm': 'BCBimax',
            'minr': minr,
            'minc': minc,
            'MaxBiclusters': number
        },
        'RowxNumber': row_matrix,
        'NumberxCol': col_matrix,
        'Number': len(raw)
    }

