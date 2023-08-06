#include <assert.h>
#include <core/umb_common.h>
#include <stdlib.h>


template<typename T> class umb_darray {
  public:
  umb_darray() {
    m_data = (T*)malloc(sizeof(T) * DEFAULT_CAP);
    m_len  = 0;
    m_cap  = DEFAULT_CAP;
  }

  void push(T& val) {
    if (m_len == DEFAULT_CAP) {
      m_cap *= 2;
      m_data = (T*)realloc(m_data, sizeof(T) * m_cap);
    }
    m_data[m_len++] = val;
  }

  T& get(size_t idx) {
    assert(idx >= 0 && idx < m_len);
    return m_data[idx];
  }

  void pop() {
    assert(m_len > 0);
    m_len--;
  }

  void reset() {
    m_len = 0;
  }

  void len() {
    return len;
  }

  void release() {
    m_len = 0;
    m_cap = 0;
    free(m_data);
  }

  private:
  T*                      m_data;
  size_t                  m_len;
  size_t                  m_cap;
  static constexpr size_t DEFAULT_CAP = 32;
};
